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
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import numpy as np

import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class CopaExample(object):
    """A single training/test example for the Copa dataset."""

    def __init__(self,
                 copa_id,
                 premise,
                 question,
                 choice0,
                 choice1,
                 label=None):
        self.copa_id = copa_id
        self.premise = premise
        self.question = question
        self.choices = [
            choice0,
            choice1,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "copa_id: {}".format(self.copa_id),
            "premise: {}".format(self.premise),
            "question: {}".format(self.question),
            "choice1: {}".format(self.choices[0]),
            "choice2: {}".format(self.choices[1]),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label
                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class CopaProcessor():
    """Processor for the COPA data set (SuperGLUE version)."""

    def get_train_examples(self, data_dir,start,end):
        """See base class."""
        return self._read_copa_examples(os.path.join(data_dir, "train.jsonl"),start,end, 'train')

    def get_dev_examples(self, data_dir,start,end):
        """See base class."""
        return self._read_copa_examples(os.path.join(data_dir, "val.jsonl"),start,end, 'val')

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._read_copa_examples(os.path.join(data_dir, "test.jsonl"), 0,0,'test')

    def get_labels(self):
        """See base class."""
        return [0,1]

    def _read_copa_examples(self, input_file, start,end,state):
        lines = []
        for line in open(input_file, "r", encoding="utf-8"):
            line = json.loads(line)
            if state == 'train':
                # if int(line['idx']) > end  or int(line['idx']) <start:
                    lines.append(line)
            elif state == 'val':
                # if start <= int(line['idx']) <= end:
                    lines.append(line)
            else:
                lines.append(line)

        examples = [
            CopaExample(
                copa_id=line["idx"],
                premise=line["premise"],
                question=line["question"],
                label=int(line["label"]) if state!="test" else None,
                choice0 = line["choice2"] if state=="train" and int(line["label"]) == 1 else line["choice1"],
                choice1 = line["choice1"] if state=="train" and int(line["label"]) == 1 else line["choice2"],
                # choice0=line["choice1"],
                # choice1 = line["choice2"],
            ) for line in lines  # we skip the line with the column names
        ]

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):
        premise_tokens = tokenizer.tokenize(example.premise)

        choices_features = []
        for choice_index, choice in enumerate(example.choices):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to choice_tokens
            premise_tokens_choice = premise_tokens[:]
            choice_tokens = tokenizer.tokenize(choice)
            # Modifies `premise_tokens_choice` and `choice_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(premise_tokens_choice, choice_tokens, max_seq_length - 3)
            if example.question == "effect":
                tokens = premise_tokens_choice + [sep_token] + choice_tokens + [sep_token]
                segment_ids = [sequence_a_segment_id] * (len(premise_tokens_choice) + 1) + [sequence_b_segment_id] * (
                            len(choice_tokens) + 1)
            else:
                tokens = choice_tokens + [sep_token] + premise_tokens_choice + [sep_token]
                segment_ids = [sequence_a_segment_id] * (len(choice_tokens) + 1) + [sequence_b_segment_id] * (
                            len(premise_tokens_choice) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        features.append(
            InputFeatures(
                example_id=example.copa_id,
                choices_features=choices_features,
                label=label
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "copa":
        return {"acc": simple_accuracy(labels, preds)}
    else:
        raise KeyError(task_name)


processors = {
    "copa": CopaProcessor,
}

output_modes = {
    "copa": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "copa": 2,
}
