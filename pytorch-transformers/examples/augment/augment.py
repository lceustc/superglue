"""
    data augmentation for superglue task.

    1. EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks
    https://arxiv.org/abs/1901.11196

    2. back translation

    3. Unsupervised data augmentation
    https://arxiv.org/pdf/1904.12848.pdf

"""
import os
import json
import copy
import string
import argparse

from eda import eda, synonym_replacement


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str, help="input file of unaugmented data")
    parser.add_argument("--task", required=True, type=str, help="task")
    parser.add_argument("--exp_name", required=True, type=str, help="experiment name.")

    ## params for eda
    parser.add_argument("--do_eda", action='store_true',
                        help="whether to do EDA augmentation.")
    parser.add_argument("--do_synonym_replacement", action='store_true',
                        help="only do synonym replacement, instead of standard EDA.")
    parser.add_argument("--num_aug", type=int, default=9,
                        help="number of augmented sentences per original sentence")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="percent of words in each sentence to be changed")

    ## params for task-specific
    parser.add_argument("--aug_passage", action='store_true',
                        help="whether to do augmentation on passage for mr tasks.")
    parser.add_argument("--sent_len_min", type=int, default=10,
                        help="Minimum length to do augmentation in a passage.")
    parser.add_argument("--aug_question", action='store_true',
                        help="whether to do augmentation on question for mr tasks.")

    return parser.parse_args()


def gen_eda(data, task, alpha, num_aug, args):
    augmented_data = []

    for i, d in enumerate(data):

        augmented_d = [d for _ in range(num_aug)]  # copy

        if args.task == 'boolq':
            passage, question = d['passage'], d['question']

            if args.aug_passage:
                """
                    augmentation on passage; since passage is composed of multiple sentences, solutions are:
                    1. split passage into multiple sentences
                        2.1 choose one sentence, and do augmentation
                        2.2 done, or repeat for all sentences
                    3. concatenate sentences back into one passage.

                    still, some problems need to be firstly fixed:
                    - lowercase? 
                    - concatenation error, some spaces are ignored
                    - 'a, b, c, d ' could not be processed since it has punctuations in betweens
                    - ...

                    but fortuantely, there samples is assigned to pseudo labels instead of some hard labels.
                """
                # split
                sentences, punctuations = split_into_sentences(passage)
                # choose qualified sentence
                idxs = [i for i, sentence in enumerate(sentences) if len(sentence.split()) >= args.sent_len_min]
                # augment
                augmented_ss = [sentences for _ in range(num_aug)]  # copy
                for _, idx in enumerate(idxs):
                    augmented_s = eda(sentences[idx], alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha,
                                      num_aug=num_aug)
                    for i, s in enumerate(augmented_s):
                        augmented_ss[i][idx] = s
                # merge and modify
                for i, ss in enumerate(augmented_ss):
                    augmented_d[i]["passage"] = merge_into_passage(ss, punctuations)

            if args.aug_question:
                """
                    augmentation on question
                """
                augmented_questions = eda(question, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha,
                                          num_aug=num_aug)
                # modify
                for i, q in enumerate(augmented_q):
                    augmented_d[i]["question"] = q

        else:
            raise NotImplementedError("%s is not implemented." % task)

        augmented_data += augmented_d

    print("generated augmented sentences with eda for " + task + " with num_aug=" + str(num_aug))
    return augmented_data


def split_into_sentences(passage):
    sentences, punctuations = [], []
    sentence = ''

    passage = list(passage)
    while passage:

        c = passage.pop(0)

        if c in string.punctuation:
            sentences.append(sentence)
            sentence = ''
            punctuations.append(c)
        else:
            sentence += c

    if sentence:
        sentences.append(sentence)

    return sentences, punctuations


def merge_into_passage(sentences, punctuations):
    passage = ''
    if len(sentences) == len(punctuations):
        for i, sentence in enumerate(sentences):
            passage += sentence + punctuations[i]
    else:
        for i, sentence in enumerate(sentences[:-1]):
            passage += sentence + punctuations[i]
        passage += sentence
    return passage


def gen_synonym_replacement(data, task, alpha, num_aug, args):
    augmented_data = []

    for i, d in enumerate(data):

        augmented_d = [copy.deepcopy(d) for _ in range(num_aug)]  # copy

        if args.task == 'boolq':
            passage, question = d['passage'], d['question']

            if args.aug_passage:
                """
                """
                _passage, n = passage.split(), int(len(passage.split()) * alpha)
                augmented_p = [" ".join(synonym_replacement(_passage, n)) for _ in range(num_aug)]

                # modify
                for i, p in enumerate(augmented_p):
                    augmented_d[i]["passage"] = p

            if args.aug_question:
                """
                    augmentation on question
                """
                augmented_questions = eda(question, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha,
                                          num_aug=num_aug)
                # modify
                for i, q in enumerate(augmented_questions):
                    augmented_d[i]["question"] = q

        else:
            raise NotImplementedError("%s is not implemented." % task)

        augmented_data += augmented_d

    print("generated augmented sentences with eda for " + task + " with num_aug=" + str(num_aug))
    return augmented_data


def main():
    args = get_config()

    ## get data
    with open(args.data, 'r') as reader:
        data = [json.loads(line.strip()) for line in reader]

    augmented_data = []

    ## get standard eda data
    if args.do_eda:
        augmented_data += gen_eda(data, args.task, args.alpha, args.num_aug, args)

    ## get synonym_replacement, without distorting the structure of texts
    if args.do_synonym_replacement:
        augmented_data += gen_synonym_replacement(data, args.task, args.alpha, args.num_aug, args)

    ## get back-translation data

    ## possibly downsample augmented data, or other choices
    data = augmented_data + data

    ## define output path and write
    from os.path import dirname, basename, join
    output_dir = join(dirname(args.data), 'eda_' + args.exp_name)
    if not os.path.isdir(output_dir): os.mkdir(output_dir)
    output_path = join(output_dir, basename(args.data))

    with open(output_path, 'w') as writer:
        for d in data:
            writer.write(json.dumps(d) + '\n')


if __name__ == "__main__":
    main()


