# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import gc
import json
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (compute_metrics, convert_examples_to_features, output_modes, processors,
                        generate_results, generate_results_for_ensemble)

from utils_common import ExponentialMovingAverage, AdversarialTraining, EntropyLoss, KLDivLoss, CrossEntropyLoss, \
    CrossEntropyLossSoft, create_soft_labels

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)),
    ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """
        Train the model.

        fix using fp16 to accelerate training.
    """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    """
        setup knowledge distilling schedules.
    """
    if args.kd_config:
        ## setup schedule of knowledge distilliation weight
        kd_weight = args.kd_weight
        ## linearly decrease the weights of soft labels
        kd_weight_decaying = kd_weight / t_total if args.kd_weight_decaying else 0
        ## temperature
        kd_temperature = args.kd_temperature if args.kd_temperature > 0. else 1.
    else:
        kd_weight, kd_weight_decaying, kd_temperature = 0, 0, 1.
    """
        ExponentialMovingAverage.
    """
    if 0 < args.ema_decay < 1:  # to dismiss EMA, set ema_decay to -1
        ema = ExponentialMovingAverage(decay=args.ema_decay)
        ema.register(model)
        logger.info("We use Exponential Moving Averagein this setting, with decay as %f" % args.ema_decay)
    else:
        ema = None
        logger.info("We do not use Exponential Moving Averagein.")
    """
        learning rate layer decay, hard coding for BERT, XLNet and RoBERTa.
    """

    def extract_n_layer(n, max_n_layer=-1):
        n = n.split('.')
        try:
            idx = n.index("layer")
            n_layer = int(n[idx + 1]) + 1
        except:
            if any(nd in n for nd in ["embeddings", "word_embedding", "mask_emb"]):
                n_layer = 0
            else:
                n_layer = max_n_layer
        return n_layer

    # we acquire the max_n_layer from inference,
    # we leave the sequence_summary layer and logits layer own same learning rate scale 1.
    # the lower 24 encoder layers shave decaying learning rate scare decay_scale ** (24-layer), layer ~ (0,23)
    max_n_layer = max([extract_n_layer(n) for n, p in model.named_parameters()]) + 1
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []  # group params by layers and weight_decay params.
    for n_layer in range(max_n_layer + 1):
        #### n_layer and decay
        optimizer_grouped_parameters.append({
            'params': [p for n, p in model.named_parameters() if (
                    extract_n_layer(n, max_n_layer=max_n_layer) == n_layer and not any(nd in n for nd in no_decay))],
            'weight_decay': args.weight_decay,
            'lr_decay': args.lr_layer_decay ** (max_n_layer - n_layer)
        })
        #### n_layer and no_decay
        optimizer_grouped_parameters.append({
            'params': [p for n, p in model.named_parameters() if (
                    extract_n_layer(n, max_n_layer=max_n_layer) == n_layer and any(nd in n for nd in no_decay))],
            'weight_decay': 0.0,
            'lr_decay': args.lr_layer_decay ** (max_n_layer - n_layer)
        })
        # #### debug info
        # ns = [n for n, _ in model.named_parameters() if (
        #     extract_n_layer(n, max_n_layer=max_n_layer) == n_layer and not any(nd in n for nd in no_decay))]
        # lr_decay = args.lr_layer_decay ** (max_n_layer-n_layer)
        # print(ns)
        # print(lr_decay)
        # print('\n\n')
    ## setting optimizer, plan to add RADAM & LookAhead
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if 0. < args.warmup_proportion < 1.0:
        warmup_steps = t_total * args.warmup_proportion
    else:
        warmup_steps = args.warmup_steps
    ## TODO: add other scheduler
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    """
        The following settings follow apex's instruction
        ``
        model.cuda() # Cuda-ing your model should occur before the call to amp.initialize
        model, optimizer = amp.initialize(model, optimizer)
        model = nn.DataParallel(model)
        ``
        https://github.com/NVIDIA/apex/issues/227
    """
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d, warmup steps = %d", t_total, warmup_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    ## set loss module. use kldivloss to replace CrossEntropyLossSoft
    cross_entropy_loss = CrossEntropyLoss()
    entropy_loss, kld_loss = EntropyLoss(), KLDivLoss(reduction='mean')
    # adversarial training hyperparameters
    ad = AdversarialTraining(eps_ad=args.adversarial_training_eps, eps_vad=args.virtual_adversarial_training_eps,
                             eta=args.virtual_adversarial_training_eta)

    # set flag
    is_first = True

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            ## model inputs
            inputs = batch[:3]
            inputs = tuple(t.to(args.device) for t in inputs)
            inputs = {'input_ids': inputs[0],
                      'attention_mask': inputs[1],
                      'token_type_ids': inputs[2] if args.model_type in ['bert',
                                                                         'xlnet'] else None, }  # XLM and RoBERTa don't use segment_ids
            if args.task_name in ["copa-ranking"]:
                #### ranking
                pass

            else:
                #### sequence classification

                ## handle unlabelled examples and knowledge distillation
                ## 1. normal examples with standard supervised training; len(batch) == 4 and not args.kd_config
                ## 2. no labels, unsupervised examples; for doing virtual adversarial training; len(batch) == 3
                ## 3. normal knowledge distilling; len(batch) == 5 and args.kd_config
                ## 4. knowledge ditilling on augmented data; len(batch) == 4 and args.kd_config
                labels, soft_labels = None, None
                if (args.kd_config and len(batch) == 5) or (not args.kd_config and len(batch) == 4):
                    labels = batch[3].to(args.device)
                if args.kd_config:
                    soft_labels = batch[3].to(args.device) if len(batch) == 4 else batch[4].to(args.device)
                ## save batchs for adversarial training
                ## does adversarial training conflic with knowledge distilling?
                if args.gradient_accumulation_steps > 1 and (step + 1) % args.gradient_accumulation_steps != 1:
                    # not the first step in accumulation_steps
                    is_first = False
                    batchs.append([inputs, labels, soft_labels])
                else:
                    # the first step in accumulation_steps
                    is_first = True
                    batchs = [[inputs, labels, soft_labels]]
                """
                    move loss calcualtion outside from module. In multi-gpu setting, DataParallel concatenate 
                    the results on batch size automatically.
                    ---------------------------------------------------------------------
                    1. knowledge distilling. final_loss = kd_wight * kd_loss + (1-kd_weight) * loss
                    reference:
                    2. objectives = ML + (adversarial training) + (entropy minimization) + (virtual adversarial training)
                    labelled data and unlablled data. for labelled data using full objectives, for unlabelled data, using 
                    (entropy minimization) + (virtual adversarial training). 
                    reference: https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_46.pdf
                    3. a special training schedule should be added to improve training.
                    reference: https://arxiv.org/pdf/1904.12848.pdf

                    ---------------------------------------------------------------------
                    TODO: move this block into optimizer module to be compatible with FP16
                    UPDATE: previous results showing that only using entropy minimization in default setting seems to destroy performance.
                """
                logits = model(**inputs)[0]
                logits = logits.view(-1, args.num_labels)
                if labels is not None or soft_labels is not None:
                    ## weighted average of normal loss and distilling loss
                    loss = 0
                    if labels is not None:
                        loss += (1 - kd_weight) * cross_entropy_loss(logits, labels.view(-1))
                    if soft_labels is not None:
                        ## Important: logits are assumed to represented as log_softmax with temperature
                        loss += kd_weight * kld_loss(F.log_softmax(logits / kd_temperature, dim=-1), soft_labels)
                    ## loss accumulation and fp16.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    tr_loss += loss.item()

                # # accumulate entropy minimization loss, not backward to influence cross entropy loss
                # if args.entropy_minimization_weight > 0.:
                #     em_loss = entropy_loss(logits) if is_first else em_loss + entropy_loss(logits)

                # virtual adversarial training need record logits of original weights
                if args.virtual_adversarial_training_weight > 0.:
                    vad_logits = [logits] if is_first else vad_logits + [logits]

            if (step + 1) % args.gradient_accumulation_steps == 0:

                # adversarial training
                if args.adversarial_training_weight > 0. and (labels is not None or soft_labels is not None):
                    # disturb model's embedding params
                    ad.disturb(model, is_virtual=False)
                    # accumulate gradient_accumulation_steps sub-batchs
                    for (inputs, labels, soft_labels) in batchs:
                        ad_loss = 0
                        logits = model(**inputs)[0]  # same inputs with slightly modified model
                        ## copy from loss calculation
                        if labels is not None:
                            ad_loss += (1 - kd_weight) * cross_entropy_loss(logits.view(-1, args.num_labels),
                                                                            labels.view(-1))
                        if soft_labels is not None:
                            ad_loss += kd_weight * kld_loss(F.log_softmax(logits / kd_temperature, dim=-1), soft_labels)
                        ad_loss = ad_loss * args.adversarial_training_weight
                        if args.gradient_accumulation_steps > 1:
                            ad_loss = ad_loss / args.gradient_accumulation_steps
                        if args.fp16:
                            with amp.scale_loss(ad_loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            ad_loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    # restore model's embedding params
                    ad.restore(model)

                # # entropy minimization
                # if args.entropy_minimization_weight > 0.:
                #     # we accumulate loss in previous steps.
                #     em_loss.backward()

                # virtual adversarial training
                if args.virtual_adversarial_training_weight > 0.:
                    # Here, we use a pseudo model to get the vad_logits
                    # pseudo_model = copy.deepcopy(model)
                    # pseudo_model.eval()

                    ## save gradient in here and zero_grad model
                    ad.save_grads(model)
                    pseudo_model = model

                    ad.add_noise(pseudo_model)  # add noise to model

                    for (inputs, _, _), logits in zip(batchs, vad_logits):
                        noise_logits = pseudo_model(**inputs)[0]
                        noise_loss = kld_loss(noise_logits, logits.data)
                        noise_loss = noise_loss * args.virtual_adversarial_training_weight
                        if args.gradient_accumulation_steps > 1:
                            noise_loss = noise_loss / args.gradient_accumulation_steps
                        if args.fp16:
                            with amp.scale_loss(noise_loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            noise_loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    ad.restore(pseudo_model)
                    ad.disturb(pseudo_model)  # re-disturb using grads from logits
                    #
                    # final adv loss
                    for inputs, _, _ in batchs:

                        logits = model(**inputs)[0]  # logits from original model
                        logits_disturb = pseudo_model(**inputs)[0]  # logits from pseudo model

                        vad_loss = kld_loss(logits, logits_disturb.data)  # get adv loss

                        if args.gradient_accumulation_steps > 1:
                            vad_loss = vad_loss / args.gradient_accumulation_steps
                        if args.fp16:
                            with amp.scale_loss(vad_loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            vad_loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    # remove pseudo model immediately
                    # del pseudo_model; gc.collect()

                    ## restore the gradient of previous steps and add on current model
                    ad.restore_grads(model)

                # update
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                model.zero_grad()
                global_step += 1
                kd_weight = kd_weight - kd_weight_decaying  # update kd_weight by kd_weight_decaying

                # EMA update
                if ema is not None: ema.update(model)

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        # before evaluation and after evaluation, cover and restore ema weights
                        if ema is not None: ema.cover(model)
                        # evaluate results, if using ema, cover the weights
                        results = evaluate(args, model, tokenizer, evaluation=True, prefix=str(global_step))
                        # restore wegihts if using ema
                        if ema is not None: ema.recover(model)

                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    if args.local_rank in [-1, 0] and args.save_steps > 0 and results['acc'] >=0.87:
                        # Save ema model checkpoint
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(global_step,results['acc']))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        if ema is not None: ema.cover(model)

                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)

                        if ema is not None: ema.recover(model)

                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if ema is not None:
        # before exit training, use ema to cover the final model weights.
        logger.info("Cover the final model params using EMA weights.")
        ema.cover(model)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, evaluation=True, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):

        if evaluation:
            # evaluation
            split = "train" if args.do_gen_kd else "dev"
            eval_dataset, eval_examples = load_and_cache_examples(args, eval_task, tokenizer, split=split)
        else:
            # testing
            eval_dataset, eval_examples = load_and_cache_examples(args, eval_task, tokenizer, split="test")

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        scores = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM and RoBERTa don't use segment_ids
                          'labels': batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if scores is None:
                scores = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                scores = np.append(scores, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        # softmax score, for further processing (ensemble)
        if args.do_gen_kd:
            scores = np.exp(scores / args.kd_temperature)  # use temperature to soft logits
        else:
            scores = np.exp(scores)
        scores = scores / scores.sum(-1, keepdims=True)

        # get scores or resutls depending on the task
        guids, candidates = [eg.guid for eg in eval_examples], None
        if eval_task in ["mnli", "mnli-mm", "mrpc", "sst-2", "qqp", "qnli", "rte", "wnli", "boolq", "wic", "wsc",
                         "multirc", "cb"]:
            preds = np.argmax(scores, axis=1)
        elif args.task_name in ["copa", "race", "cnn", "cbtest"]:
            preds = scores[:, 1]
        elif args.task_name == "record":
            preds, candidates = scores[:, 1], [eg.answer_text for eg in eval_examples]
        else:
            raise "Not implement evaluation metrics for the task %s" % args.task_name
        scores = scores[:, 1]  # only get logits for true class

        if evaluation:
            #### Run evaluation
            result = compute_metrics(eval_task, preds, out_label_ids, guids=guids, candidates=candidates)
            #
            result["eval_loss"] = eval_loss / nb_eval_steps
            results.update(result)
            # report resutls
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            if args.task_name in ["cnn", "cbtest", "race"]:
                # we ignore these tasks
                continue

            # write evaluation results for visualization
            output_eval_file = os.path.join(eval_output_dir, prefix + "_eval_results.jsonl")
            generate_results(args.task_name, preds, eval_examples, output_eval_file, out_label_ids=out_label_ids)
            # write evaluation results for check ensemble / knowledge distilling
            output_eval_file_ckpt = os.path.join(eval_output_dir, prefix + "_eval_results_ckpt.jsonl")
            generate_results_for_ensemble(output_eval_file_ckpt, guids, scores, out_label_ids)
        else:
            #### Run testing
            # for single submission file
            output_test_file = os.path.join(eval_output_dir, eval_task + "_test_results.jsonl")
            generate_results(args.task_name, preds, eval_examples, output_test_file)
            # for ensemble submission file
            output_test_file_ckpt = os.path.join(eval_output_dir, eval_task + "_test_results_ckpt.jsonl")
            generate_results_for_ensemble(output_test_file_ckpt, guids, scores)

    return results


def load_and_cache_examples(args, task, tokenizer, split="train"):
    if args.local_rank not in [-1, 0] and split != "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        split,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    #### UPDATE: we load examples in any case
    label_list = processor.get_labels()
    if split == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif split == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    elif split == "test":
        examples = processor.get_test_examples(args.data_dir)
    else:
        raise Exception("split must in [train, dev, test]")

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                args.task_name,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ['roberta']),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            # torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    if args.kd_config and split == "train":
        # provided soft logits from knowledge distillation config file
        soft_labels = create_soft_labels(args.kd_config)
        soft_labels = torch.tensor(soft_labels, dtype=torch.float) * args.kd_weight
        # valid checking, same number of examples
        assert len(soft_labels) == len(features)
        logger.info("Using soft labels to do knowledge distilling.")
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, soft_labels)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset, examples  # also return examples for post-processing


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    ## adapt to specific task
    parser.add_argument("--ignore_logits_layer", action='store_true',
                        help="whether to skip initialization of logits layers.")
    parser.add_argument("--ignore_sequence_summary_layer", action='store_true',
                        help="whether to skip initialization of sequence summary layers.")
    ## load averaged checkpoint, need set do_train False, do_eval True, and eval_all_checkpoints False
    parser.add_argument("--load_averaged_checkpoint", action='store_true',
                        help="whether to load averaged checkpoint from a directory of training checkpoints.")

    ## single task
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## generate soft-labels with temperature
    parser.add_argument("--kd_temperature", default=1., type=float,
                        help="Control the temperature of softmax in generating logits files.")
    ## knowledge distillation training
    parser.add_argument("--kd_config", default=None, type=str,
                        help="Config file for knowledge distillation, if provided, do a knowledge distilling training.")
    parser.add_argument("--kd_weight", default=0.5, type=float,
                        help="Summation weight for knowledge distillation.")
    ## kd_weight scheduling
    parser.add_argument("--kd_weight_decaying", action='store_true',
                        help="Wether let knowledge distilling weights decaying linearly, and reach 0 finally.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the test set.")
    parser.add_argument("--do_gen_kd", action='store_true',
                        help="Whether to generate knowledge distillation file.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    ## uncased model
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    ## learning rate difference between original BertAdam and now paramters.
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    ## learning rate layer decay
    parser.add_argument("--lr_layer_decay", default=1.0, type=float,
                        help="layer learning rate decay.")

    ## training tricks implemented, including (virtual) adversarial training, entropy minimization and entropy maximization (unproven trick)
    # adversarial training
    parser.add_argument("--adversarial_training_weight", default=0., type=float,
                        help="loss weight for adversarial_training, set 0. to disable.")
    parser.add_argument("--adversarial_training_eps", default=1.0, type=float,
                        help="hyperparameters for adversarial_training, control perturbation of gradient.")
    # entropy minimization
    parser.add_argument("--entropy_minimization_weight", default=0., type=float,
                        help="loss weight for entropy_minimization, set 0. to disable, on both labelled data and unlabelled data.")
    # virtual adversarial training, tricky and unproven.
    parser.add_argument("--virtual_adversarial_training_weight", default=0., type=float,
                        help="loss weight for virtual_adversarial_training, set 0. to disable, on both labelled data and unlabelled data.")
    parser.add_argument("--virtual_adversarial_training_eps", default=1.0, type=float,
                        help="hyperparameters for virtual adversarial_training.")
    parser.add_argument("--virtual_adversarial_training_eta", default=0.1, type=float,
                        help="hyperparameters for virtual adversarial_training.")
    # exponential moving average; effectively in boolq and multirc task.
    parser.add_argument("--ema_decay", default=0.999, type=float,
                        help="hyperparameters to control EMA updating, set -1 to disable.")

    ## optimizer params.
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")

    ##
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # add file handler to log training info
    if not os.path.isdir(args.output_dir): os.mkdir(args.output_dir)
    fh = logging.FileHandler(os.path.join(args.output_dir, "training.log"))
    logger.addHandler(fh)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE/SuperGLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    args.num_labels = num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    """
        add param: 
            ignore_logits_layer:            whether to ignore the logits layer
            ignore_sequence_summary_layer:  whether to ignore the sequence_summary_layer
            load_averaged_checkpoint:       whether load a model with averaged weights
            ...
    """
    logger.info("Training/evaluation parameters %s", args)
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.do_train:
        # load model
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            ignore_logits_layer=args.ignore_logits_layer,
                                            ignore_sequence_summary_layer=args.ignore_sequence_summary_layer,
                                            load_averaged_checkpoint=args.load_averaged_checkpoint)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        train_dataset, _ = load_and_cache_examples(args, args.task_name, tokenizer, split="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # testing/evaluation
    if (args.do_eval or args.do_test or args.do_gen_kd) and args.local_rank in [-1, 0]:
        if args.do_train:
            # training and then evaluation.
            model_dir = args.output_dir
        else:
            # just evaluation or generating kd files, use the assigned model directory.
            model_dir = args.model_name_or_path
        # generate evaluation files for all checkpoints in the model_dir
        checkpoints = [model_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(model_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        # evaluation
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, load_averaged_checkpoint=args.load_averaged_checkpoint)
            model.to(args.device)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)
            if args.do_test:
                evaluate(args, model, tokenizer, evaluation=False)
            if args.do_gen_kd:
                evaluate(args, model, tokenizer, evaluation=True, prefix=global_step)
            if args.do_eval:
                result = evaluate(args, model, tokenizer, evaluation=True, prefix=global_step)
                result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    return


if __name__ == "__main__":
    main()
