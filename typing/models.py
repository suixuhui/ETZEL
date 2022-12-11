import torch.nn as nn
import torch
from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)
from torch.nn.init import xavier_uniform_ as xavier_uniform


class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()

        self.name = args.model
        self.gpu = args.gpu
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.types = torch.load(args.DATA_DIR + "/crowd/types.pt").cuda(self.gpu)
        self.final = nn.Linear(self.types.size(1), 10331)
        xavier_uniform(self.final.weight)

    def forward(self, inputs_id, segment_ids, masks, labels, mode):

        x, out_pooler = self.bert(inputs_id, segment_ids, attention_mask=masks)
        alpha = torch.nn.functional.softmax(torch.matmul(self.types, x.transpose(1, 2)), dim=2)
        m = alpha.matmul(x)
        logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        if mode != "generate":
            loss = self.loss(logits, labels.float())
            return logits, loss
        else:
            return logits


def pick_model(args):
    if args.model == "bert":
        model = Bert(args)
    else:
        raise RuntimeError("wrong model name")
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model