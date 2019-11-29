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
import json
from io import open
from collections import defaultdict, Counter
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a

class COPAExample(object):
    def __init__(self,mtc_id,premise,question,choice0,choice1,label=None):
        self.mtc_id=mtc_id
        self.premise=premise
        self.question=question
        self.choices = [
            choice0,
            choice1,
        ]
        self.label=label

class MtcExample(object):
    def __init__(self,mtc_id,premise,question,choice0,choice1,choice2=None,choice3=None,label=None,):
        self.mtc_id=mtc_id
        self.premise=premise
        self.question=question
        self.choices = [
            choice0,
            choice1,
            choice2,
            choice3,
        ]
        self.label=label


class MultiChoiceInputFeatures(object):
    def __init__(self,example_id,choices_features,label):
        self.example_id = example_id
        self.choice_features = [
            {
                'input_ids' : input_ids,
                'input_mask' : input_mask,
                'segment_ids' : segment_ids
            }for _,input_ids,input_mask,segment_ids in choices_features
        ]
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_jsonl(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(json.loads(line.strip()))
            return lines

    @classmethod
    def _read_csv(cls,input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class PretrainProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return None

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "val")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line["idx"])
            text_a = line["text"]
            examples.append(
                InputExample(guid=guid, text_a=text_a))
        return examples



class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "val")

    def get_labels(self):
        """See base class."""
        # return ["entailment", "not_entailment"]
        return ["not_entailment" , "entailment"]

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line["idx"])
            text_a = line["premise"]
            text_b = line["hypothesis"]
            if set_type in ["train", "val"]:
                label = line["label"]
            else:
                label = "entailment"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,task_name,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True,
                                 is_mtc=False,
                                 ):

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 2
        if len(tokens_a) > max_seq_length - special_tokens_count:
            tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * (len(input_ids))

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

        # tokens = ["[CLS]"] + tokens + ["[SEP]"]
        # input_ids = [tokenizer.convert_tokens_to_ids("[CLS]")] + input_ids + [tokenizer.convert_tokens_to_ids("[SEP]")]


        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          ))
        if ex_index < 1 :
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(len(input_ids))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(len(input_mask))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(len(segment_ids))
    return features


class CbProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "val")

    def get_labels(self):
        """See base class."""
        return ["entailment", "contradiction", "neutral"]

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line["idx"])
            text_a = line["premise"]
            text_b = line["hypothesis"]
            if set_type in ["train", "val"]:
                label = line["label"]
            else:
                label = "entailment"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


class COPAProcessor(DataProcessor):
    """Processor for the COPA data set (SUPERGLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "btest.jsonl")), "val")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines:
            examples.append(
                MtcExample(mtc_id=line["idx"],
                            premise=line["premise"],
                            label=int(line["label"]) if set_type is not "test" else None,
                            choice0=line["choice1"],
                            choice1=line["choice2"],
                            question=line["question"]
                )
            )
        return examples

class SWAGProcessor(DataProcessor):
    """Processor for the COPA data set (SUPERGLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "val.csv")), "val")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i ==0:
                continue

            examples.append(
                MtcExample(mtc_id=line[2],
                            premise=line[4],
                            question=line[5],
                            label=int(line[-1]) if set_type is not "test" else None,
                            choice0=line[7],
                            choice1=line[8],
                            choice2=line[9],
                            choice3=line[10],
                )
            )
        return examples

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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average='binary'):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {
        "acc": acc,
        "f1": f1,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)

    if task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "cb":
        return acc_and_f1(preds, labels, average="macro")
    elif task_name == "copa":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "swag":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "pretrain":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


processors = {
    "rte": RteProcessor,
    "cb": CbProcessor,
    "copa": COPAProcessor,
    "swag":SWAGProcessor,
    "pretrain":PretrainProcessor,
}

output_modes = {
    "rte": "classification",
    "cb": "classification",
    "copa": "classification",
    "swag":"classification",

}

GLUE_TASKS_NUM_LABELS = {
    "rte": 2,
    "cb": 3,
    "copa": 2,
    "swag":4,
}

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choice_features
        ]
        for feature in features
    ]