# -*- coding : utf-8 -*-
import torch
from torch import nn

from allennlp.nn.util import batched_index_select
from allennlp.modules import FeedForward

from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

from copy import deepcopy

# from empn_models.evolutionary_mask_pointer_network import EMPN_for_Entity
from empn_models.pnc_network import PNC_for_ALL


class Span_based_Encoder(BertPreTrainedModel):
    def __init__(self, model_config, encoder_config, ner_label_count):
        super().__init__(model_config)
        self.decoder_config = encoder_config

        self.bert_model = BertModel(model_config)
        if encoder_config.fix_bert_embeddings:
            self.bert_model.embeddings.word_embeddings.weight.requires_grad = False
            self.bert_model.embeddings.position_embeddings.weight.requires_grad = False
            self.bert_model.embeddings.token_type_embeddings.weight.requires_grad = False
        self.dropout_layer = nn.Dropout(model_config.hidden_dropout_prob)
        self.span_width_embedding = nn.Embedding(encoder_config.max_span_length + 1, encoder_config.width_embedding_dim)
        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=model_config.hidden_size * 2 + encoder_config.width_embedding_dim,
                        num_layers=2,
                        hidden_dims=encoder_config.hidden_dim_for_NER,
                        activations=torch.nn.ReLU(),
                        dropout=0.2),
            nn.Linear(encoder_config.hidden_dim_for_NER, ner_label_count)
        )

        self.decoder_projection1 = nn.Linear(model_config.hidden_size * 2 + encoder_config.width_embedding_dim,
                                             model_config.hidden_size)
        self.decoder_projection2 = nn.Linear(model_config.hidden_size, model_config.hidden_size)

        # 补充内容
        self.feature_concat_projection = nn.Linear(2 * model_config.hidden_size, model_config.hidden_size)
        decoder_config_copy = deepcopy(model_config)
        decoder_config_copy.add_cross_attention = True
        decoder_config_copy.is_decoder = True
        decoder_config_copy.num_hidden_layers = 2  # 减半，正常是12
        self.decoder_encoder = BertEncoder(decoder_config_copy)
        self.entity_pointer_network = PNC_for_ALL(model_config.hidden_size,
                                                  model_config.hidden_size,
                                                  self.decoder_encoder,
                                                  length_info=5)

        self.ner_linear_layer = nn.Linear(model_config.hidden_size, ner_label_count)

        self.init_weights()

    @staticmethod
    def masked_average_pooling(sequence, mask):
        normalized_mask = mask.masked_fill(mask == 0, -1e9).float()
        weights = torch.softmax(normalized_mask, -1)
        return torch.matmul(weights.unsqueeze(1), sequence).squeeze(1)

    def forward(self, input_tokens, input_attention_mask, span_indices):
        span_embeddings, encoder_output, pooled_features, rl_loss = \
            self.get_span_embeddings(input_tokens, input_attention_mask, span_indices)

        classifier_outputs = []
        hidden = span_embeddings
        if self.decoder_config.NER_mode == "2":
            for layer in self.ner_classifier:
                hidden = layer(hidden)
                classifier_outputs.append(hidden)
            ner_logits = classifier_outputs[-1]

        hidden = span_embeddings
        relation_embeddings = self.decoder_projection2(torch.relu(self.decoder_projection1(hidden)))

        pooled_features = pooled_features.unsqueeze(1)
        bf_embeddings = torch.cat((pooled_features, relation_embeddings), dim=1)
        rd_embeddings = relation_embeddings

        if self.decoder_config.NER_mode == "1":
            ner_logits = self.ner_linear_layer(torch.relu(relation_embeddings))

        if self.decoder_config.cross_attention_mode_for_BF == "2":
            bf_encoder_output = bf_embeddings
        else:
            bf_encoder_output = encoder_output

        if self.decoder_config.cross_attention_mode_for_RD == "2":
            rd_encoder_output = rd_embeddings
        else:
            rd_encoder_output = encoder_output

        return ner_logits, bf_embeddings, rd_embeddings, \
               bf_encoder_output, rd_encoder_output, rl_loss

    def get_span_embeddings(self, input_tokens, input_attention_mask, span_indices):
        sequence_output, pooled_output = self.bert_model(input_ids=input_tokens, attention_mask=input_attention_mask,
                                                         return_dict=False)
        sequence_output = self.dropout_layer(sequence_output)

        # 增加指针过程
        selected_states, _, rl_loss = self.entity_pointer_network(
            x=sequence_output,
            pooled=pooled_output,
            context_masks=input_attention_mask,
        )
        pooled_selected = self.masked_average_pooling(selected_states, input_attention_mask)
        pooled_output = self.feature_concat_projection(torch.cat([pooled_output, pooled_selected], -1))
        sequence_output = self.feature_concat_projection(
            torch.cat([sequence_output, selected_states], -1))

        span_start_indices = span_indices[:, :, 0].view(span_indices.size(0), -1)
        span_start_embeddings = batched_index_select(sequence_output, span_start_indices)

        span_end_indices = span_indices[:, :, 1].view(span_indices.size(0), -1)
        span_end_embeddings = batched_index_select(sequence_output, span_end_indices)

        span_widths = span_indices[:, :, 2].view(span_indices.size(0), -1)
        width_embeddings = self.span_width_embedding(span_widths)

        span_embeddings = torch.cat((span_start_embeddings, span_end_embeddings,
                                     width_embeddings), dim=-1)
        return span_embeddings, sequence_output, pooled_output, rl_loss