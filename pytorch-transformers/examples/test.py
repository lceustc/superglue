from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,RobertaForMultipleChoice,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

tokenizer = BertTokenizer.from_pretrained("/data/lce/pretrained_model/bert", do_lower_case=False)
a= "今天不滑又心痒痒"

b = tokenizer.tokenize(a)
print(b)