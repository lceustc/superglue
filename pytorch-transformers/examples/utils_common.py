import os
import gc
import glob
import json
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import KLDivLoss, CrossEntropyLoss

import numpy as np

class ExponentialMovingAverage:
    """Exponential Moving Average implementation to improve stability
    """

    def __init__(self, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self, model):
        """"""
        module = get_module(model)

        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """"""
        module = get_module(model)

        for name, param in module.named_parameters():
            if name in self.shadow:
                new_average = (1.0 - self.decay) * param.data.clone() + self.decay * self.shadow[name]
                self.shadow[name] = new_average

    def cover(self, model):
        """cover current weights to do evaluation."""
        self.backup = {}

        module = get_module(model)
        # store a copy
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data

        # cover model using shadow weights
        for name, param in module.named_parameters():
            if name in self.shadow:
                param.data = self.shadow[name].clone()

    def recover(self, model):
        """"""
        module = get_module(model)

        for name, param in module.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        #
        self.backup = {};
        gc.collect()

class AdversarialTraining(object):
    """
        only store gradient, remove replace/inplcae operation
    """
    DISTURB_WEIGHTS = ['embeddings']
    # DISTURB_WEIGHTS = ['word_embeddings']
    BIAS = 1e-8


    def __init__(self, eps_ad=1., eps_vad=1., eta=0.1):
        self.backup_data = {}
        self.backup_noise = {}
        self.eps_ad = eps_ad
        self.eps_vad = eps_vad
        self.eta = eta
        ##
        self.grads = {}

    def restore(self, model,sigma=None):
        if sigma is not None:
            for name, param in model.named_parameters():
                if "word_embedding" in name:
                    param.data.add_(-sigma)
            del param,sigma;import gc;gc.collect();torch.cuda.empty_cache()
        else:
            for name, param in model.named_parameters():
                if name in self.backup_data:
                    param.data.add_(-self.backup_data[name])
                if name in self.backup_noise:
                    param.data.add_(-self.backup_noise[name])
            self.backup_data = {}
            self.backup_noise = {}
            del name,param;import gc;gc.collect();torch.cuda.empty_cache()


    def disturb(self, model, is_virtual=False,sigma=None):
        eps = self.eps_vad if is_virtual else self.eps_ad
        for name, param in model.named_parameters():
            if param.requires_grad and any([n in name for n in self.DISTURB_WEIGHTS]):
                if sigma is not None:
                    param.data.add_(sigma)
                    # self.backup_data[name] = sigma
                    del sigma;import gc;gc.collect();torch.cuda.empty_cache()
                else:
                    norm = torch.norm(param.grad, dim=-1, keepdim=True).sum()
                    r = eps * param.grad / (norm + self.BIAS)  # bias smaller than 1e-4 cause problem when using fp16
                    param.data.add_(r)
                    self.backup_data[name] = r
        del name, param;import gc;gc.collect();torch.cuda.empty_cache()

    def add_noise(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and any([n in name for n in self.DISTURB_WEIGHTS]):
                noise = self.eta * torch.rand(param.data.size(), dtype=torch.float, device=param.device)
                param.data.add_(noise)
                self.backup_noise[name] = noise

    def save_grads(self, model):
        """save model's gradient"""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grads[name] = param.grad.data
        model.zero_grad()

    def restore_grads(self, model):
        """restore model's gradient"""
        # ## debug
        # for key, val in self.grads.items():
        #     print(key, val.shape, val)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.grads:
                param.grad.data += self.grads[name]
        self.grads = {}


class EntropyLoss(nn.Module):
    """Entropy minimization to expand difference between two logits.
    """

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, logits):
        e = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        e = -1.0 * e.sum(dim=-1).mean()  # sum over num_classes and mean over per sample
        return e


class CrossEntropyLossSoft(nn.Module):

    def __init__(self):
        super(CrossEntropyLossSoft, self).__init__()

    def forward(self, logits, soft_label):
        ces = soft_label * F.log_softmax(logits, dim=-1)
        ces = -1.0 * ces.sum(dim=-1).mean()  # sum over num_classes and mean over per sample
        return ces


class Mixout(object):
    def __init__(self, pmodel_path, p=0.7):
        self.p = p
        self.param = {}
        self.old_param = {}
        static_dic = torch.load(pmodel_path+"/pytorch_model.bin")
        for k, v in static_dic.items():
            if "classifier" not in k:
                self.old_param[k] = v
        del static_dic;
        import gc;
        gc.collect();
        torch.cuda.empty_cache()

    def restore(self, model):
        for k, v in model.named_parameters():
            if k in self.param.keys():
                v.data.mul_(0)
                v.data.add_(self.param[k])
        self.param = {}

    def replace(self, model):
        for k, v in model.named_parameters():
            if k in self.old_param.keys():
                self.param[k] = v
                p_new = torch.zeros_like(v) + (1-self.p)
                new_bernoulli = torch.bernoulli(p_new)
                old_bernoulli = torch.ones_like(v) - new_bernoulli
                v.data.mul_(new_bernoulli)
                v.data.add_(old_bernoulli*self.old_param[k].cuda())


def get_module(model):
    # handle multi-gpu setting
    return model.module if hasattr(model, "module") else model


def check_idxs(list_idxs):
    return all([idx == list_idxs[0] for idx in list_idxs[1:]])


#### Knowledge Distillition
def create_soft_labels(config_file):
    """
        score_dir:    the directory for collecting soft logits.
    """
    config = json.load(open(config_file))  # load config
    task, paths, num_models = config['task'], config['paths'], len(config['paths'])
    weights = config.get('weights', np.ones(num_models) / num_models)

    # exclude wsc
    assert task in ['boolq', 'multirc', 'record', 'wic', 'rte', 'copa', 'cb']

    # load ckpt results
    predictions = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as reader:
            ckpt_results = [json.loads(line.strip()) for line in reader]  # load examples
            predictions.append(ckpt_results)
    # transfer from [num_ckpt, num_results] to [num_results, num_ckpt]
    predictions = list(zip(*predictions))
    #
    guids, results = [], []
    for i, prediction in enumerate(predictions):
        assert check_idxs([pred['idx'] for pred in prediction])  # valid checking
        example_result = np.zeros(num_models)
        for i, pred in enumerate(prediction):
            example_result[i] = pred['score']
        # guids
        guids.append(pred['idx'])
        # aggrate resutls
        example_result = np.average(example_result, weights=weights)
        assert 0. <= example_result <= 1., "averaged logits exceed range [0, 1]."
        #
        results.append([1 - example_result, example_result])  #

    return results
