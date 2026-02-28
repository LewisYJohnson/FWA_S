import torch
import torch.nn as nn

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """
        Attention layer
    """

    def __init__(self, hidden_size, first_layer_size, **kwargs):
        super(AttentionLayer, self).__init__()
        self.query_transform = nn.Linear(first_layer_size * hidden_size, hidden_size, bias=False)
        self.key_transform = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_transform = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_states, decoder_state, attention_mask=None):
        projected_query = self.query_transform(encoder_states)
        projected_key = self.key_transform(decoder_state)
        combined_features = torch.tanh(projected_query + projected_key)
        attention_scores = self.value_transform(combined_features)
        if attention_mask is not None:
            attention_mask = torch.clamp(attention_mask, 0, 1)
            attention_mask = attention_mask.unsqueeze(-1)
            attention_scores += (attention_mask * -1e9)
        attention_weights = attention_scores.softmax(1)
        attention_weights = attention_weights.squeeze(-1)
        return attention_weights


class SequenceDecoder(nn.Module):
    """
        Decoder class for PointerLayer
    """

    def __init__(self, hidden_size, bert_decoder, **kwargs):
        super(SequenceDecoder, self).__init__()
        self.bert_decoder = bert_decoder

    def forward(self, current_input, previous_state):
        combined_state = torch.cat([previous_state, current_input], 1)
        decoder_output = self.bert_decoder(combined_state)['last_hidden_state'][:, -1, :]
        return decoder_output

    def get_initial_state(self, input_data):
        return None


class PNC_for_ALL(nn.Module):
    def __init__(self, base_dim, hidden_dim, bert_decoder, max_sequence_length=7):
        super(PNC_for_ALL, self).__init__()
        self.attention_mechanism = AttentionLayer(hidden_dim, first_layer_size=1)
        self.sequence_decoder = SequenceDecoder(hidden_dim, bert_decoder, name="sequence_decoder")
        self.max_sequence_length = max_sequence_length
        self.feature_projector = nn.Linear(base_dim, hidden_dim)

        self.gating_mechanism = nn.Linear(hidden_dim, hidden_dim)
        self.inverse_decoder = SequenceDecoder(hidden_dim, bert_decoder, name="inverse_decoder")

    def forward(self, input_sequence, pooled_features=None, context_mask=None, entity_mask=None):
        context_mask = 1 - context_mask
        gate_weights = torch.sigmoid(self.gating_mechanism(pooled_features))

        previous_state = torch.zeros_like(pooled_features).to(input_sequence.device)
        current_state = gate_weights * pooled_features

        inverse_previous_state = torch.zeros_like(pooled_features).to(input_sequence.device)
        inverse_current_state = (1 - gate_weights) * pooled_features

        cumulative_mask = torch.zeros_like(context_mask).to(input_sequence.device).float()
        cumulative_probabilities = torch.zeros_like(context_mask).to(input_sequence.device).float()
        selected_indices_container = []

        reinforcement_loss = None
        for step_idx in range(self.max_sequence_length):
            if step_idx == 0:
                (previous_state, current_state, cumulative_mask,
                 step_indices, cumulative_probabilities,
                 inverse_previous_state, inverse_current_state, reinforcement_loss) = \
                    self.step(previous_state, current_state, cumulative_mask,
                              cumulative_probabilities, input_sequence,
                              context_mask, inverse_previous_state, inverse_current_state)
            else:
                (previous_state, current_state, cumulative_mask,
                 step_indices, cumulative_probabilities,
                 inverse_previous_state, inverse_current_state, reinforcement_loss) = \
                    self.step(previous_state, current_state, cumulative_mask,
                              cumulative_probabilities, input_sequence,
                              context_mask, inverse_previous_state,
                              inverse_current_state, reinforcement_loss)
            selected_indices_container.append(step_indices)

        final_indices = torch.max(torch.cat([x.unsqueeze(1) for x in selected_indices_container], 1), -1)[1]
        cumulative_probs = cumulative_probabilities.unsqueeze(-1)
        selected_sequence = cumulative_probs * input_sequence
        return selected_sequence, final_indices, reinforcement_loss

    def step(self, prev_state, curr_state, cumulative_mask, cumulative_probs,
             input_seq, context_mask, inv_prev_state, inv_curr_state, rl_loss=None):

        decoder_output = self.sequence_decoder(curr_state.unsqueeze(1), prev_state.unsqueeze(1))
        decoder_expanded = decoder_output.unsqueeze(1).repeat(1, input_seq.shape[1], 1)
        attention_probs = self.attention_mechanism(decoder_expanded, input_seq, mask=context_mask + cumulative_mask)

        inverse_output = self.inverse_decoder(inv_curr_state.unsqueeze(1), inv_prev_state.unsqueeze(1))
        inverse_expanded = inverse_output.unsqueeze(1).repeat(1, input_seq.shape[1], 1)
        inverse_probs = self.attention_mechanism(inverse_expanded, input_seq, mask=context_mask + cumulative_mask)

        base_probs = torch.zeros_like(attention_probs).to(input_seq.device)
        max_indices = torch.max(attention_probs, dim=-1)[1].unsqueeze(-1)
        base_probs.scatter_(dim=-1, index=max_indices, value=1)

        inverse_base = torch.zeros_like(inverse_probs).to(input_seq.device)
        sorted_indices = torch.argsort(inverse_probs, dim=-1, descending=False)
        restore_indices = torch.argsort(sorted_indices, dim=-1, descending=False).to(torch.float32)
        threshold_val = (torch.max(restore_indices) - 1).to(torch.float32)
        inverse_base = torch.where(restore_indices >= threshold_val,
                                   1.0 * (restore_indices - threshold_val),
                                   torch.zeros(restore_indices.size(), device=restore_indices.device,
                                               dtype=restore_indices.dtype))

        selected_features = input_seq * (attention_probs + base_probs).unsqueeze(-1)
        selected_features = selected_features.max(dim=1)[0]
        step_probs = (attention_probs + base_probs) + cumulative_probs

        inverse_features = input_seq * (inverse_probs + inverse_base).unsqueeze(-1)
        inverse_features = inverse_features.max(dim=1)[0]

        attention_mask = torch.ge(attention_probs, base_probs.float()).reshape(base_probs.shape).float()
        attention_mask = torch.ones_like(attention_mask) - attention_mask

        inverse_mask = torch.ge(inverse_probs, inverse_base.float()).reshape(inverse_base.shape).float()
        inverse_mask = torch.ones_like(inverse_mask) - inverse_mask

        combined_mask = attention_mask + inverse_mask
        combined_mask = torch.clamp(combined_mask, 0, 1)

        prob_difference = 1 / (torch.exp(attention_probs - inverse_probs))
        step_rl_loss = torch.sum(base_probs * prob_difference)

        if rl_loss is not None:
            rl_loss = rl_loss + step_rl_loss
            return (decoder_output, selected_features, combined_mask + cumulative_mask,
                    combined_mask, step_probs, inverse_output, inverse_features, rl_loss)
        else:
            return (decoder_output, selected_features, combined_mask + cumulative_mask,
                    combined_mask, step_probs, inverse_output, inverse_features, step_rl_loss)