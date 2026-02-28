# -*- coding : utf-8 -*-
import torch
from torch import nn
from allennlp.nn.util import batched_index_select
from .Bert_Minimal import Bert_for_NEC

from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers import BertTokenizer, BertPreTrainedModel
import sys
sys.path.append("..")


class NEC_Decoder(BertPreTrainedModel):
    def __init__(self, model_config, decoder_config):
        super().__init__(model_config)
        self.decoder_config = decoder_config
        self.model_config = model_config

        self.embedding_layer = BertEmbeddings(model_config)
        if decoder_config.fix_bert_embeddings:
            self.embedding_layer.word_embeddings.weight.requires_grad = False
            self.embedding_layer.position_embeddings.weight.requires_grad = False
            self.embedding_layer.token_type_embeddings.weight.requires_grad = False

        self.encoder_module = Bert_for_NEC(model_config, decoder_config=decoder_config)

        self.span_projection1 = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        self.span_projection2 = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        self.span_scoring = nn.Linear(model_config.hidden_size, 1, bias=False)

        self.init_weights()

        torch.nn.init.orthogonal_(self.span_projection1.weight, gain=1)
        torch.nn.init.orthogonal_(self.span_projection2.weight, gain=1)

    def forward(self, input_attention_mask, span_attention_mask, relation_logits, encoder_output,
                question_tokens, question_attention_mask, mask_positions, answer_labels,
                token_type_ids):

        if self.decoder_config.cross_attention_mode_for_BF == "2":
            encoder_mask = span_attention_mask
        else:
            encoder_mask = input_attention_mask

        extended_encoder_mask = self.generate_extended_mask(encoder_mask)
        extended_question_mask = self.generate_extended_mask(question_attention_mask)

        embedded_question = self.embedding_layer(input_ids=question_tokens, token_type_ids=token_type_ids)

        hidden_representations = embedded_question
        question_output = self.encoder_module(hidden_representations, extended_question_mask, encoder_output,
                                              extended_encoder_mask)
        # question_output.shape = [bz, question_seq_len, model_config.hidden_size(768)]
        answer_scores = self.compute_answer_scores(relation_logits, question_output, mask_positions)
        return answer_scores

    def generate_extended_mask(self, attention_mask):
        if attention_mask.dim() == 3:
            extended_mask = attention_mask[:, None, :, :]
        else:
            extended_mask = attention_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0
        return extended_mask

    def compute_answer_scores(self, relation_embeddings, question_output, mask_positions):
        masked_embeddings = batched_index_select(question_output, mask_positions)
        if self.decoder_config.answer_attention == "add":
            answer_scores = self.span_scoring(torch.tanh(
                self.span_projection1(masked_embeddings).unsqueeze(2) +
                self.span_projection2(relation_embeddings).unsqueeze(1))).squeeze(-1)
        else:
            answer_scores = torch.bmm(
                torch.tanh(self.span_projection1(masked_embeddings)),
                torch.tanh(self.span_projection2(relation_embeddings)).transpose(1, 2))
        return answer_scores





