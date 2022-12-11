from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)
from pytorch_transformers.tokenization_bert import BertTokenizer
import torch
TYPE_TITLE_TAG = "[unused3]"
titles = []
descriptions = []
with open("../data/crowd/types_definition.txt", 'rt') as f:
    for line in f.readlines():
        line = line.strip()
        title_description = line.split(" <sep> ")
        title = title_description[0].replace("_", " ")
        titles.append(title)
        if len(title_description) > 1 and title_description[1] != "TODO":
            description = title_description[1]
        else:
            description = ""
        descriptions.append(description)

max_seq_length = 80
gpu = 1
bert = BertModel.from_pretrained("bert-base-uncased")
bert = bert.cuda(gpu)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
cls_token = tokenizer.cls_token
sep_token = tokenizer.sep_token


def encoding(titles, descriptions):
    categories_pooler = None
    for title, description in zip(titles, descriptions):
        title_tokens = tokenizer.tokenize(title)
        type_tokens = title_tokens
        if description != "":
            description_tokens = tokenizer.tokenize(description)
            type_tokens = title_tokens + [sep_token] + description_tokens
        type_tokens = type_tokens[: max_seq_length - 2]
        type_tokens = [cls_token] + type_tokens + [sep_token]

        input_ids = tokenizer.convert_tokens_to_ids(type_tokens)
        masks = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        masks += padding
        input_ids, masks = torch.LongTensor(input_ids).cuda(gpu), torch.LongTensor(masks).cuda(gpu)
        input_ids, masks = torch.unsqueeze(input_ids, dim=0), torch.unsqueeze(masks, dim=0)
        with torch.no_grad():
            output_bert, output_pooler = bert(input_ids, attention_mask=masks)
        output_pooler = output_pooler.cpu()
        if categories_pooler == None:
            categories_pooler = output_pooler
        else:
            categories_pooler = torch.cat((categories_pooler, output_pooler))
    return categories_pooler

types_pooler = encoding(titles, descriptions)
torch.save(types_pooler, "../data/crowd/types.pt")