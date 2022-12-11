import numpy as np
import json
import torch.nn as nn
import math

from pytorch_transformers.tokenization_bert import BertTokenizer
ENT_TITLE_TAG = "[unused2]"


def prepare_instance_bert(filename, args, max_length):
    instances = []
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir, do_lower_case=True)
    datas = []
    all_types = get_all_types(args.DATA_DIR + "/crowd/types.txt")
    with open(filename) as f:
        for line in f.readlines():
            datas.append(json.loads(line))

    for i in range(len(datas)):
        mention = datas[i]["mention"]
        context = datas[i]["sentence"]
        labels = datas[i]["labels"] + datas[i]["fine_labels"] + datas[i]["ultra_fine_labels"]

        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        context_tokens = tokenizer.tokenize(context)
        mention_tokens = tokenizer.tokenize(mention)
        mention_tokens = [cls_token] + mention_tokens + [sep_token]
        segment_ids = [0] * len(mention_tokens)
        context_tokens = mention_tokens + context_tokens + [sep_token]
        segment_ids += [1] * (len(context_tokens) - len(mention_tokens))

        context_tokens = context_tokens[: max_length]
        segment_ids = segment_ids[: max_length]

        input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
        masks = [1] * len(input_ids)
        padding = [0] * (max_length - len(input_ids))
        input_ids += padding
        masks += padding
        segment_ids += padding

        labels_idx = np.zeros(10331)
        for label in labels:
            for j, la in enumerate(all_types):
                if la == label:
                    labels_idx[j] = 1

        dict_instance = {'tokens_id': input_ids,
                         'segment_ids': segment_ids,
                         'masks': masks,
                         'labels': labels_idx}

        instances.append(dict_instance)

    return instances



def prepare_instance_generate(filename, args, max_length):
    instances = []
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir, do_lower_case=True)
    datas = []
    all_types = get_all_types(args.DATA_DIR + "/crowd/types.txt")
    with open(filename, mode='r', encoding="utf-8") as f:
        for line in f:
            datas.append(json.loads(line.strip()))

    for i in range(len(datas)):
        mention = datas[i]["title"]
        context = datas[i]["text"]
        document_id = datas[i]["document_id"]


        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        context_token = tokenizer.tokenize(context)
        mention_tokens = tokenizer.tokenize(mention)
        mention_tokens = [cls_token] + mention_tokens + [sep_token]
        if len(context_token) > (max_length - len(mention_tokens)):
            for j in range(math.ceil(len(context_token) / (max_length - len(mention_tokens)))):
                segment_ids = [0] * len(mention_tokens)
                if (j + 1) * (max_length - len(mention_tokens)) < len(context_token):
                    context_tokens = mention_tokens + context_token[j * (max_length - len(mention_tokens)): (j + 1) * (max_length - len(mention_tokens))] + [sep_token]
                else:
                    context_tokens = mention_tokens + context_token[j * (max_length - len(mention_tokens)): len(context_token)] + [sep_token]
                segment_ids += [1] * (len(context_tokens) - len(mention_tokens))

                context_tokens = context_tokens[: max_length]
                segment_ids = segment_ids[: max_length]

                input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
                masks = [1] * len(input_ids)
                padding = [0] * (max_length - len(input_ids))
                input_ids += padding
                masks += padding
                segment_ids += padding


                dict_instance = {'tokens_id': input_ids,
                                 'segment_ids': segment_ids,
                                 'masks': masks,
                                 "document_id": document_id}

                instances.append(dict_instance)
        else:
            segment_ids = [0] * len(mention_tokens)
            context_tokens = mention_tokens + context_token + [sep_token]
            segment_ids += [1] * (len(context_tokens) - len(mention_tokens))

            context_tokens = context_tokens[: max_length]
            segment_ids = segment_ids[: max_length]

            input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
            masks = [1] * len(input_ids)
            padding = [0] * (max_length - len(input_ids))
            input_ids += padding
            masks += padding
            segment_ids += padding

            dict_instance = {'tokens_id': input_ids,
                             'segment_ids': segment_ids,
                             'masks': masks,
                             "document_id": document_id}

            instances.append(dict_instance)

    return instances


from torch.utils.data import Dataset
class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def my_collate_bert(x):
    inputs_id = [x_['tokens_id'] for x_ in x]
    segment_ids = [x_['segment_ids'] for x_ in x]
    masks = [x_['masks'] for x_ in x]
    labels = [x_['labels'] for x_ in x]
    # general_labels = [x_['general_labels'] for x_ in x]
    # fine_labels = [x_['fine_labels'] for x_ in x]
    # ufine_labels = [x_['ufine_labels'] for x_ in x]

    return inputs_id, segment_ids, masks, labels


def my_collate_generate(x):
    inputs_id = [x_['tokens_id'] for x_ in x]
    segment_ids = [x_['segment_ids'] for x_ in x]
    masks = [x_['masks'] for x_ in x]
    document_id = [x_['document_id'] for x_ in x]

    return inputs_id, segment_ids, masks, document_id

def my_collate_generate_mention(x):
    inputs_id = [x_['tokens_id'] for x_ in x]
    segment_ids = [x_['segment_ids'] for x_ in x]
    masks = [x_['masks'] for x_ in x]
    document_id = [x_['document_id'] for x_ in x]

    return inputs_id, segment_ids, masks, document_id


def get_all_types(filename):
    all_types = []
    with open(filename, 'rt') as f:
        for line in f.readlines():
            line = line.strip()
            all_types.append(line)
    return all_types


def get_output_index(outputs, threshold=0.5):
  pred_idx = []
  sigmod = nn.Sigmoid()
  outputs = sigmod(outputs).data.cpu()
  for single_dist in outputs:
    single_dist = single_dist.numpy()
    arg_max_ind = np.argmax(single_dist)
    pred_id = [arg_max_ind]
    pred_id.extend(
      [i for i in range(len(single_dist)) if single_dist[i] > threshold and i != arg_max_ind])
    pred_idx.append(pred_id)
  return pred_idx

def f1(p, r):
  if r == 0.:
    return 0.
  return 2 * p * r / float(p + r)

def strict(true_and_prediction):
  num_entities = len(true_and_prediction)
  correct_num = 0.
  for true_labels, predicted_labels in true_and_prediction:
    correct_num += set(true_labels) == set(predicted_labels)
  precision = recall = correct_num / num_entities
  return precision, recall, f1(precision, recall)

def macro(true_and_prediction):
  num_examples = len(true_and_prediction)
  p = 0.
  r = 0.
  pred_example_count = 0.
  pred_label_count = 0.
  gold_label_count = 0.
  precision = 0.
  recall = 0.
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
      pred_label_count += len(predicted_labels)
      per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
      p += per_p
    if len(true_labels):
      gold_label_count += 1
      per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
      r += per_r
  if pred_example_count > 0:
    precision = p / pred_example_count
  if gold_label_count > 0:
    recall = r / gold_label_count
  avg_elem_per_pred = pred_label_count / pred_example_count
  return num_examples, pred_example_count, avg_elem_per_pred, precision, recall, f1(precision, recall)

def micro(true_and_prediction):
  num_examples = len(true_and_prediction)
  num_predicted_labels = 0.
  num_true_labels = 0.
  num_correct_labels = 0.
  pred_example_count = 0.
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
    num_predicted_labels += len(predicted_labels)
    num_true_labels += len(true_labels)
    num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
  if pred_example_count == 0:
    return num_examples, 0, 0, 0, 0, 0
  precision = num_correct_labels / num_predicted_labels
  recall = num_correct_labels / num_true_labels
  avg_elem_per_pred = num_predicted_labels / pred_example_count
  return num_examples, pred_example_count, avg_elem_per_pred, precision, recall, f1(precision, recall)