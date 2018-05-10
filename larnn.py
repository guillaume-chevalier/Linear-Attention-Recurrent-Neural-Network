
"""PyTorch implementation of the LARNN, by Guillaume Chevalier."""

import math
import copy
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F

from multi_head_attention import MultiHeadedAttention, PositionalEncoding


__author__ = "Guillaume Chevalier"
__license__ = "MIT License"
__copyright__ = "Copyright 2018, Guillaume Chevalier"


class LARNN(nn.Module):
    def __init__(self, input_size, hidden_size, attention_heads, num_layers, larnn_window_size,
                 larnn_mode='residual', use_positional_encoding=True, activation_on_keys_and_values=True,
                 is_stacked_residual=False, device="cuda", dropout=0.0):
        """A LARNN which can contain stacked LARNN Cells, similar to an LSTM.

        Args:
            input_size: number of features in the input `x`
            hidden_size: number of features in the inner LSTM's state
            attention_heads: the count 'h' of attention heads
            num_layers: number of stacked layers in depth
            larnn_window_size: how far back in time does the attention sees.
            larnn_mode='concat|residual|layer': how the attention works in the
                LARNNCell. See documentation of the LARNNCell for more details.
            use_positional_encoding: wheter or not to use geometric series
                of sines and cosines for positional encoding, re-generated
                at each attention step
            is_stacked_residual: wheter or not the stacked LARNN layers are
                added to the value of the first LARNN layer (and using a final
                batch norm), for `num_layers` times
            device: string conaining "cuda" or "cpu".
            dropout: how much dropout is applied on the output of the cell
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.larnn_window_size = larnn_window_size
        self.larnn_mode = larnn_mode
        self.use_positional_encoding = use_positional_encoding
        self.activation_on_keys_and_values = activation_on_keys_and_values
        self.device = device
        self.dropout = dropout

        self.larnn_cells = [
            LARNNCell(input_size, hidden_size, attention_heads, larnn_window_size,
                      larnn_mode, use_positional_encoding, activation_on_keys_and_values, dropout).to(device)
            for _ in range(num_layers)]

        self.num_layers = num_layers
        self.is_stacked_residual = is_stacked_residual

        self.init_parameters()
        self.to(device)

    def init_parameters(self):
        for param in self.parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)

    def forward(self, input, state=None):
        if state is None:
            # If not using Truncated BPTT, init a new inner cell every time:
            batch_size = input.size(1)
            state = self.num_layers * [LARNNCellState(
                batch_size,
                self.hidden_size,
                self.larnn_window_size,
                self.use_positional_encoding,
                self.device)]

        # Stacking the layers:
        new_state = []
        hidden = input
        for i, (_cell, _state) in enumerate(zip(self.larnn_cells, state)):
            _out, _state = self._forward_cell(self.larnn_cells[0], input, state[0])

            if self.is_stacked_residual:
                hidden = hidden + _out
            else:
                hidden = _out
            new_state.append(_state)

        output = hidden
        return output, new_state

    def _forward_cell(self, cell, input, state):
        # Loop on sequence
        outputs = []
        for i in range(input.size(0)):
            x = input[i]
            o, state = cell(x, state)
            outputs.append(o)
        output_tensor = torch.stack(outputs)
        return output_tensor, state


class LARNNCellState(nn.Module):
    def __init__(self, batch_size, hidden_size, larnn_window_size, use_positional_encoding, device):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.larnn_window_size = larnn_window_size
        self.use_positional_encoding = use_positional_encoding

        self.states = deque()
        self.states.append((
            torch.zeros([batch_size, hidden_size]).to(device),  # hidden (gated output)
            torch.zeros([batch_size, hidden_size]).to(device)  # memory cell (inner)
        ))

        if use_positional_encoding:
            # Positional Encoding in the state, used in `.get_past_cells_for_attention()`
            self.positional_encoding = PositionalEncoding(
                batch_size=batch_size, max_sequence_length=larnn_window_size, device=device)
        self.to(device)

    def update_most_recent_state(self, new_state):
        self.states.append(new_state)

        if len(self.states) > self.larnn_window_size:
            self.states.popleft()

    def get_most_recent_cell(self):
        return self.states[-1]

    def get_past_cells_for_attention(self):
        # Get the past states' inner cells
        past_cells = torch.stack([state[1] for state in self.states])  # size [sequence_length, batch_size, hidden_size]

        if self.use_positional_encoding:
            # Append positional_encoding to features (inner axis)
            cells_with_positional_encoding = self.positional_encoding(
                past_cells.transpose(0, 1)).transpose(0, 1)
            return cells_with_positional_encoding  # returned shape: [sequence_length, batch_size, hidden_size]
        else:
            return past_cells  # returned shape: [sequence_length, batch_size, hidden_size]


class LARNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, attention_heads, larnn_window_size,
                 larnn_mode='residual', use_positional_encoding=True,
                 activation_on_keys_and_values=True, dropout=0.0):
        """A LARNN Cell on which it's possible to loop as an LSTM Cell.

        Args:
            input_size: number of features in the input `x`
            hidden_size: number of features in the inner LSTM's state
            attention_heads: the count 'h' of attention heads.
            larnn_window_size: how far back in time does the attention sees.
            larnn_mode='concat|residual|layer': how the attention is plugged
                into the inner LSTM's layers. Specifically:
                  - 'residual' will add the attention result to `Wx*x+Wh*h+b`
                    such as to get `Wx*x+Wh*h+Wa*a+b`.
                  - 'layer' will create a new layer such as
                    doing `Wa*(concat(x, h, a))+b`.
                Note:
                    `a(K, Q, V) = MultiHead(softmax(Q*K'/sqrt(dk))*V)` like in Attention Is All You Need (AIAYN).
                    `Q = Wq*(Wx*x + Wh*h + bias1)`
                    `K = V = Wk*(a "larnn_window_size" number of most recent cells)`
            use_positional_encoding: wheter or not to use geometric series
                of sines and cosines for positional encoding, re-generated
                at each attention step
            dropout: how much dropout is applied on the output
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.larnn_window_size = larnn_window_size
        self.larnn_mode = larnn_mode
        self.use_positional_encoding = use_positional_encoding
        self.activation_on_keys_and_values = activation_on_keys_and_values
        self.attention_heads = attention_heads
        self.dropout = dropout
        assert hidden_size % attention_heads == 0, "'hidden_size' must be divisible by 'attention_heads'."

        self.input_to_hidden = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.hidden_to_hidden = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.batch_norm_pre_activation = torch.nn.BatchNorm1d(4 * hidden_size)
        self.batch_norm_post_activation = torch.nn.BatchNorm1d(hidden_size)
        self.batch_norm_attention_result = torch.nn.BatchNorm1d(hidden_size)

        self.input_and_hidden_to_query = nn.Linear(input_size + hidden_size, hidden_size, bias=True)

        nb_positional_features = int(use_positional_encoding) * PositionalEncoding.get_features_dimensionnality(larnn_window_size)
        if larnn_mode == 'residual':
            # Attention will be added to Wx and Wh as `Wx*x + Wh*h + Wa*a + bias1`.
            self.attention_to_cell = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        elif larnn_mode == 'layer':
            # Attention will be post-processed like `Wa*(concat(x, h, a)) + bias2`
            self.attention_to_cell = nn.Linear(3 * hidden_size, 4 * hidden_size, bias=True)

        self.multi_headed_attention = MultiHeadedAttention(
            attention_heads, hidden_size + nb_positional_features,
            hidden_size, activation_on_keys_and_values, dropout=0.1)

        self.init_parameters("pytorch_default")

    def init_parameters(self, style="xavier_uniform"):
        if style == "xavier_uniform":
            for param in self.parameters():
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
        elif style == "pytorch_default":
            invsqrt = 1.0 / math.sqrt(self.hidden_size)
            for param in self.parameters():
                param.data.uniform_(-invsqrt, invsqrt)

    def forward(self, input, state):
        # Unpack arguments:
        previous_state = state.get_most_recent_cell()
        prev_hidden, prev_cell = previous_state

        # LARNN's Linear Attention:
        pre_activation = self.linear_attention(input, prev_hidden, state)
        # replacing the previous line with the following one would be an LSTM not a LARNN:
        # pre_activation = self.input_to_hidden(input) + self.hidden_to_hidden(prev_hidden)

        # Classic LSTM functions:
        input_values = pre_activation[:, :self.hidden_size].tanh()
        packed_gates = pre_activation[:, self.hidden_size:].sigmoid()
        forget_gate = packed_gates[:, :self.hidden_size]
        input_gate = packed_gates[:, self.hidden_size:2 * self.hidden_size]
        cell = torch.mul(input_values, input_gate) + torch.mul(prev_cell, forget_gate)
        output_gate = packed_gates[:, -self.hidden_size:]
        hidden = torch.mul(output_gate, F.elu(cell))  # elu instead of tahn
        hidden = self.batch_norm_post_activation(hidden)  # specially, batch norm here

        # Bundle for output:
        if self.training and self.dropout > 0.0:
            F.dropout(hidden, p=self.dropout, training=self.training, inplace=True)
        current_state = (hidden, cell)
        state.update_most_recent_state(current_state)

        return hidden, state

    def linear_attention(self, x, h, state):
        prev_cells = state.get_past_cells_for_attention()  # shape (larnn_window_size, batch_size, hidden_size)

        # `V = K = Wk*(a "larnn_window_size" number of most recent cells)`
        values = V = K = prev_cells  # shape (larnn_window_size, batch_size, hidden_size)

        # `Q = Wxh*concat(x, h) + bxh`
        ih = torch.cat([x, h], -1)  # Concat on features
        query = Q = F.elu(self.input_and_hidden_to_query(ih))  #   # shape (batch_size, hidden_size)

        # `a(K, Q, V) = MultiHeadSoftmax(Q*K'/sqrt(dk))*V` like in Attention Is All You Need (AIAYN).
        query = query.unsqueeze(1)  # wants [batch_size, 1, hidden_size]
        values = values.transpose(0, 1)  # wants [batch_size, larnn_window_size, hidden_size]
        attention = F.elu(self.multi_headed_attention(query, values, values))  # attention result is [batch_size, 1, hidden_size]
        attention = self.batch_norm_attention_result(attention.squeeze())  # wants [batch_size, hidden_size]

        if self.larnn_mode == 'residual':
            # Attention will be added to Wx and Wh as `Wx*x + Wh*h + Wa*a + b`.
            Wx_Wh_Wa_b = self.input_to_hidden(x) + self.hidden_to_hidden(h) + self.attention_to_cell(attention)
            pre_activation = Wx_Wh_Wa_b
        elif self.larnn_mode == 'layer':
            # Attention will be post-processed like `Wa*(concat(x, h, a)) + b`
            Wxha_b = torch.cat([x, h, attention], -1)  # Concat on features
            pre_activation = self.attention_to_cell(Wxha_b)
        else:
            raise ValueError("'larnn_mode' must take the string value 'residual' or 'layer', not {}".format(self.larnn_mode))

        preac = self.batch_norm_pre_activation(pre_activation)
        return pre_activation


if __name__ == '__main__':
    """Debug the model."""

    for use_positional_encoding in [False, True]:
        for is_stacked_residual in [False, True]:
            for larnn_mode in ['residual', 'layer']:

                print(
                    "use_positional_encoding, is_stacked_residual, larnn_mode:",
                    use_positional_encoding, is_stacked_residual, larnn_mode)

                time_steps = 128
                batch_size = 64
                input_size = 32
                hidden_size = 32

                larnn = LARNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    attention_heads=8,
                    num_layers=2,
                    larnn_window_size=10,
                    larnn_mode=larnn_mode,
                    use_positional_encoding=use_positional_encoding,
                    is_stacked_residual=is_stacked_residual,
                    dropout=0.2
                )
                X_train = torch.autograd.Variable(torch.rand((time_steps, batch_size, input_size)))  # , requires_grad=True)

                larnn.train()
                _output, _hidden = larnn(X_train)

                print(_output.size())
                print("")

                del larnn
                del _output
                del _hidden
