import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from pytorch_transformers import  PretrainedConfig,BertPreTrainedModel

class MappingLang(nn.Module):
    def __init__(self):
        super(MappingLang,self).__init__()
        self.vocab_size1=21128
        self.vocab_size2=30522
        self.interdim=128

        self.embedding1 = nn.Embedding(self.vocab_size1,self.interdim)
        self.dense1 = nn.Linear(self.interdim,self.vocab_size2,bias=False)
        self.dense4 = nn.Linear(self.vocab_size2, self.interdim,bias=False)
        self.embedding4 = nn.Embedding(self.interdim, self.vocab_size1)
        self.dense4.weight.data = self.dense1.weight.data.t()
        self.embedding4.weight.data = self.embedding1.weight.data.t()

        self.embedding2 = nn.Embedding(self.vocab_size1, self.interdim)
        self.dense2 = nn.Linear(self.interdim, self.vocab_size2, bias=False)
        self.dense5 = nn.Linear(self.vocab_size2, self.interdim, bias=False)
        self.embedding5 = nn.Embedding(self.interdim, self.vocab_size1)
        self.dense5.weight.data = self.dense2.weight.data.t()
        self.embedding5.weight.data = self.embedding2.weight.data.t()

        self.embedding3 = nn.Embedding(self.vocab_size1, self.interdim)
        self.dense3 = nn.Linear(self.interdim, self.vocab_size2, bias=False)
        self.dense6 = nn.Linear(self.vocab_size2, self.interdim, bias=False)
        self.embedding6 = nn.Embedding(self.interdim, self.vocab_size1)
        self.dense6.weight.data = self.dense3.weight.data.t()
        self.embedding6.weight.data = self.embedding3.weight.data.t()

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

    def forward(self,input_ids,is_forward=True):
        if is_forward:
            emb1 = self.embedding1(input_ids[:,:-2])
            emb2 = self.embedding2(input_ids[:,1:-1])
            emb3 = self.embedding3(input_ids[:,2:])
            inter = self.dense1(emb1)+self.dense2(emb2)+self.dense3(emb3)
            one_hot = self.gumbel_softmax(inter,0.1)
            return one_hot
        else:
            #input_ids: B T |v2|
            den4 = self.dense4(input_ids)
            den5 = self.dense5(input_ids)
            den6 = self.dense6(input_ids)
            logits = torch.matmul(den4,self.embedding4.weight)+\
                     torch.matmul(den5,self.embedding5.weight)+\
                     torch.matmul(den6,self.embedding6.weight)

            ids = self.gumbel_softmax(logits,0.1)
            return logits,ids

