from pytorch_transformers import RobertaTokenizer
import json
import spacy


def _truncate_seq_pair(tokens_a, tokens_b, max_length , pos1=None ,pos2=None):
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
            if pos1 is not None:
                pos1.pop()
        else:
            tokens_b.pop()
            if pos2 is not None:
                pos2.pop()

dic = {"word": "class", "sentence1": "An emerging professional class.",
       "sentence2": "Apologizing for losing your temper, even though you were badly provoked, showed real class.",
       "idx": 0, "label": "false", "start1": 25, "start2": 85, "end1": 30, "end2": 90, "version": 1.1,
       "pos1": "NOUN", "pos2": "NOUN"}

tokenizer = RobertaTokenizer.from_pretrained("/data/lce/pretrained_model/roberta-large", do_lower_case=False)
nlp = spacy.load("en_core_web_sm")
POS_LABELS = [
    "SPACE",  # space
    "ADJ",  # adjective                  big, old, green, incomprehensible, first
    "ADP",  # adposition                 in, to, during
    "ADV",  # adverb                     very, tomorrow, down, where, there
    "AUX",  # auxiliary                  is, has (done), will (do), should (do)
    "CONJ",  # conjunction                and, or, but
    "CCONJ",  # coordinating conjunction   and, or, but
    "DET",  # determiner                 a, an, the
    "INTJ",  # interjection               psst, ouch, bravo, hello
    "NOUN",  # noun                       girl, cat, tree, air, beauty
    "NUM",  # numeral                    1, 2017, one, seventy-seven, IV, MMXIV
    "PART",  # particle                   ’s, not,
    "PRON",  # pronoun                    I, you, he, she, myself, themselves, somebody
    "PROPN",  # proper noun                Mary, John, London, NATO, HBO
    "PUNCT",  # punctuation                ., (, ), ?
    "SCONJ",  # subordinating conjunction  if, while, that
    "SYM",  # symbol                     $, %, §, ©, +, −, ×, ÷, =, :), 😝
    "VERB",  # verb                       run, runs, running, eat, ate, eating
    "X",  # other                      sfpksdpsxmsa
]
POS_MAPPING = {label_: index for index, label_ in enumerate(POS_LABELS)}
features = []
mask_p1 = []
tokens_a = []
pos_a = []
mask_p2 = []
tokens_b = []
pos_b = []
max_seq_length = 128
doc1 = nlp(dic["sentence1"])
for token in doc1:
    t = tokenizer.tokenize(token.text)
    tokens_a += t
    pos_a += [POS_MAPPING[token.pos_]] * len(t)
if 1:
    doc2 = nlp(dic["sentence2"])
    for token in doc2:
        t = tokenizer.tokenize(token.text)
        tokens_b += t
        pos_b += [POS_MAPPING[token.pos_]] * len(t)
    special_tokens_count = 3
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count,pos_a,pos_b)

tokens = tokens_a + ["[SEP]"]
mask_p1 = [1] * len(tokens_a) + [0]
mask_p2 = [0] * (len(tokens_a)+ 1)
pos_a += [0]
pos_b = [0]*(len(pos_a)) +pos_b
if 0 and tokens_b is not None:
    # roberta uses an extra separator b/w pairs of sentences
    tokens += [sep_token]
    mask_p1 += [0]
    mask_p2 += [0]
    pos_a += [0]
    pos_b = [0] +pos_b
segment_ids = [0] * len(tokens)

if tokens_b:
    tokens += tokens_b + ["[SEP]"]
    pos_a += [0] * (len(tokens_b) +1)
    pos_b += [0]
    mask_p1 += [0] * (len(tokens_b)+ 1)
    mask_p2 += [1] * len(tokens_b) + [0]
    segment_ids += [1] * (len(tokens_b) + 1)

if 0:
    tokens = tokens + [cls_token]
    mask_p2 += [0]
    mask_p1 += [0]
    pos_a += [0]
    pos_b += [0]
    segment_ids = segment_ids + [cls_token_segment_id]
else:
    tokens = ["[CLS]"] + tokens
    mask_p2 = [0] + mask_p2
    mask_p1 = [0] + mask_p1
    pos_a = [0] + pos_a
    pos_b = [0] + pos_b
    segment_ids = [0] + segment_ids


input_ids = tokenizer.convert_tokens_to_ids(tokens)

input_mask = [1 ] * len(input_ids)

padding_length = max_seq_length - len(input_ids)
if 0:
    input_ids = ([pad_token] * padding_length) + input_ids
    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
    segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    mask_p1 = [0] * padding_length + mask_p1
    mask_p2 = [0] * padding_length + mask_p2
else:
    input_ids = input_ids + (["[PAD]"] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    segment_ids = segment_ids + ([0] * padding_length)
    mask_p1 += [0] * padding_length
    mask_p2 += [0] * padding_length


print(len(pos_a))
print(pos_a)
# assert len(input_ids) == max_seq_length
# assert len(input_mask) == max_seq_length
# assert len(segment_ids) == max_seq_length
# assert len(mask_p1) == max_seq_length
# assert len(mask_p2) == max_seq_length
# assert len(pos_a) == max_seq_length
# assert len(pos_b) == max_seq_length