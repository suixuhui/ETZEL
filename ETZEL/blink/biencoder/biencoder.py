# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.init import xavier_uniform_ as xavier_uniform

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_utils import WEIGHTS_NAME, CONFIG_NAME

from blink.common.ranker_base import BertEncoder, get_model_obj
from blink.common.optimizer import get_bert_optimizer
from zeshel_utils import id_to_world
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG


from torch.nn.init import xavier_uniform_ as xavier_uniform


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params):
        super(BiEncoderModule, self).__init__()
        self.device = 0
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params['bert_model'])
        self.context_encoder = BertEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = BertEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.category_transformer1 = nn.TransformerEncoderLayer(768, nhead=8).cuda(self.device)
        self.linking_transformer1 = nn.TransformerEncoderLayer(768, nhead=8).cuda(self.device)
        self.category_transformer2 = nn.TransformerEncoderLayer(768, nhead=8).cuda(self.device)
        self.linking_transformer2 = nn.TransformerEncoderLayer(768, nhead=8).cuda(self.device)
        self.config = ctxt_bert.config
        self.final = nn.Linear(768, 10331).cuda(self.device)
        xavier_uniform(self.final.weight)
        self.final2 = nn.Linear(768, 10331).cuda(self.device)
        xavier_uniform(self.final2.weight)
    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        embedding_ctxt = None
        ctct_bert_output = None
        if token_idx_ctxt is not None:
            embedding_ctxt, ctct_bert_output = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        cands_bert_output = None
        if token_idx_cands is not None:
            embedding_cands, cands_bert_output = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands, ctct_bert_output, cands_bert_output


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = 0
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = "[CLS]"
        self.END_TOKEN = "[SEP]"
        # self.bs = params["train_batch_size"]
        self.tokenizer = BertTokenizer.from_pretrained(
            params["bert_model"], do_lower_case=params["lowercase"]
        )
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModule(self.params)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_bert_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )
 
    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        embedding_context, _, ctct_bert_output, _ = self.model(
            token_idx_cands, segment_idx_cands, mask_cands, None, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cands, self.NULL_IDX
        )
        _, embedding_cands, _, cands_bert_output = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        entity_category_transformer_out = self.model.category_transformer2(cands_bert_output)
        entity_linking_input = (entity_category_transformer_out + cands_bert_output) / 2

        entity_linking_transformer_out = self.model.linking_transformer2(entity_linking_input)
        entity_linking_hidden_last = entity_linking_transformer_out[:, 0, :]

        entity_transformer_output = entity_linking_hidden_last
        return embedding_cands.cpu().detach(), entity_transformer_output.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
        self,
        text_vecs,
        cand_vecs,
        random_negs=True,
        cand_encs=None,  # pre-computed candidate encoding.
        latent_encs=None
    ):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            text_vecs, self.NULL_IDX
        )
        embedding_ctxt, _, ctct_bert_output, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        mention_category_transformer_out = self.model.category_transformer1(ctct_bert_output)
        mention_category_hidden_last = mention_category_transformer_out[:, 0, :]
        mention_linking_input = (mention_category_transformer_out + ctct_bert_output) / 2

        mention_linking_transformer_out = self.model.linking_transformer1(mention_linking_input)
        mention_linking_hidden_last = mention_linking_transformer_out[:, 0, :]

        mention_transformer_output = mention_linking_hidden_last
        ctct_bert_output = mention_category_hidden_last

        category_logits = None
        for i in range(ctct_bert_output.size(0)):
            x = torch.unsqueeze(ctct_bert_output[i], dim=0)
            category_logit = self.model.final(x)
            if category_logits is None:
                category_logits = category_logit
            else:
                category_logits = torch.cat((category_logits, category_logit))

        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t()) + mention_transformer_output.mm(latent_encs.t()), category_logits

        # Train time. We compare with all elements of the batch
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            cand_vecs, self.NULL_IDX
        )
        _, embedding_cands, _, cands_bert_output = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )

        entity_category_transformer_out = self.model.category_transformer2(cands_bert_output)
        entity_category_hidden_last = entity_category_transformer_out[:, 0, :]
        entity_linking_input = (entity_category_transformer_out + cands_bert_output) / 2

        entity_linking_transformer_out = self.model.linking_transformer2(entity_linking_input)
        entity_linking_hidden_last = entity_linking_transformer_out[:, 0, :]

        entity_transformer_output = entity_linking_hidden_last
        cands_bert_output = entity_category_hidden_last

        entity_category_logits = None
        for i in range(cands_bert_output.size(0)):
            x = torch.unsqueeze(cands_bert_output[i], dim=0)
            category_logit = self.model.final2(x)
            if entity_category_logits is None:
                entity_category_logits = category_logit
            else:
                entity_category_logits = torch.cat((entity_category_logits, category_logit))
        if random_negs:
            return embedding_ctxt.mm(embedding_cands.t()) + mention_transformer_output.mm(entity_transformer_output.t()), category_logits, entity_category_logits
        else:
            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores, category_logits, entity_category_logits

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input, mention_label, category_label, label_input=None):
        flag = label_input is None
        scores, category_logits, cands_category_logits = self.score_candidate(context_input, cand_input, flag)
        bs = scores.size(0)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            loss = loss_fct(scores, label_input)
        category_loss_fct = nn.BCEWithLogitsLoss()
        category_loss = category_loss_fct(category_logits, mention_label.float())
        cands_category_loss = category_loss_fct(cands_category_logits, category_label.float())
        category_loss += cands_category_loss
        return loss, category_loss, scores, category_logits, cands_category_logits


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
