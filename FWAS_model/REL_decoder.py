# -*- coding : utf-8 -*-
import torch
from torch import nn

from data_preprocess.constant import scierc_rel_labels_constant, hacred_rel_labels_constant, \
    _2017t10_rel_labels_constant, ace05_rel_labels_constant, conll04_rel_labels_constant, ADE_rel_labels_constant
from .Bert_Minimal import Bert_for_REL
from data_preprocess import nyt_rel_labels_constant, webnlg_rel_labels_constant, webnlg_rel_labels_constant_no_star

import sys
sys.path.append("..")
BertLayerNorm = torch.nn.LayerNorm


class REL_Decoder(nn.Module):
    def __init__(self, model_config, decoder_config):
        super().__init__()
        self.decoder_config = decoder_config
        self.model_config = model_config
        self.encoder_module = Bert_for_REL(model_config, decoder_config=decoder_config)

        self.layer_norm = BertLayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)
        self.dropout_layer = nn.Dropout(model_config.hidden_dropout_prob)
        self.query_embedding = nn.Embedding(decoder_config.queries_num_for_RD, model_config.hidden_size)

        if decoder_config.task == "webnlg":
            if decoder_config.star == 1:
                relation_label_count = len(webnlg_rel_labels_constant) + 1
            else:
                relation_label_count = len(webnlg_rel_labels_constant_no_star) + 1
        elif decoder_config.task == "scierc":
            relation_label_count = len(scierc_rel_labels_constant) + 1
        elif decoder_config.task == "nyt":
            relation_label_count = len(nyt_rel_labels_constant) + 1
        else:
            relation_label_count = eval("len(" + decoder_config.task + "_rel_labels_constant) + 1")

        self.classifier = nn.Linear(model_config.hidden_size, relation_label_count)
        self.head_scorer1 = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        self.head_scorer2 = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        self.head_scorer3 = nn.Linear(model_config.hidden_size, 1, bias=False)

        self.tail_scorer1 = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        self.tail_scorer2 = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        self.tail_scorer3 = nn.Linear(model_config.hidden_size, 1, bias=False)

        torch.nn.init.orthogonal_(self.head_scorer1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_scorer2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_scorer1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_scorer2.weight, gain=1)
        torch.nn.init.orthogonal_(self.query_embedding.weight, gain=1)

    def forward(self, input_attention_mask, span_attention_mask, encoder_output, relation_embeddings):

        batch_size = span_attention_mask.shape[0]
        query_vectors = self.query_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_vectors = self.dropout_layer(self.layer_norm(query_vectors))

        if self.decoder_config.cross_attention_mode_for_RD == "2":
            encoder_mask = span_attention_mask
        else:
            encoder_mask = input_attention_mask

        extended_encoder_mask = self.generate_extended_mask(encoder_mask)

        decoder_output = self.encoder_module(query_vectors, encoder_output, extended_encoder_mask)

        relation_logits = self.classifier(decoder_output)
        head_scores = self.head_scorer3(torch.tanh(
            self.head_scorer1(decoder_output).unsqueeze(2) +
            self.head_scorer2(relation_embeddings).unsqueeze(1))).squeeze(-1)
        tail_scores = self.tail_scorer3(torch.tanh(
            self.tail_scorer1(decoder_output).unsqueeze(2) +
            self.tail_scorer2(relation_embeddings).unsqueeze(1))).squeeze(-1)

        head_scores = head_scores.masked_fill((1 - span_attention_mask.unsqueeze(1)).bool(), -10000.0)
        tail_scores = tail_scores.masked_fill((1 - span_attention_mask.unsqueeze(1)).bool(), -10000.0)

        return relation_logits, head_scores, tail_scores

    def generate_extended_mask(self, attention_mask):
        if attention_mask.dim() == 3:
            extended_mask = attention_mask[:, None, :, :]
        else:
            extended_mask = attention_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0
        return extended_mask








