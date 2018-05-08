
"""PyTorch implementation of the LARNN, by Guillaume Chevalier."""

import math
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F


__author__ = "Guillaume Chevalier"
__license__ = "MIT License"
__copyright__ = "Copyright 2018, Guillaume Chevalier"


class LARNN(nn.Module):
    def __init__(self, input_size, hidden_size, attention_heads, num_layers, larnn_window_size,
                 larnn_mode='residual', use_positional_encoding=True, is_stacked_residual=False, dropout=0.0):
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
            dropout: how much dropout is applied on the output of the cell
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.larnn_window_size = larnn_window_size
        self.larnn_mode = larnn_mode
        self.use_positional_encoding = use_positional_encoding
        self.dropout = dropout

        self.larnn_cells = [
            LARNNCell(input_size, hidden_size, attention_heads, larnn_window_size,
                      larnn_mode, use_positional_encoding, dropout)
            for _ in range(num_layers)]

        self.num_layers = num_layers
        self.is_stacked_residual = is_stacked_residual

    def forward(self, input, state=None):
        if state is None:
            # If not using Truncated BPTT, init a new inner cell every time:
            batch_size = input.size(1)
            state = self.num_layers * [LARNNCellState(
                batch_size,
                self.hidden_size,
                self.larnn_window_size)]

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
        output_tensor = torch.cat(outputs, 0)
        return output_tensor, state


class LARNNCellState(nn.Module):
    def __init__(self, batch_size, hidden_size, larnn_window_size):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.larnn_window_size = larnn_window_size

        self.states = deque()
        self.states.append((
            torch.zeros([1, batch_size, hidden_size]),  # hidden (gated output)
            torch.zeros([1, batch_size, hidden_size])  # memory cell (inner)
        ))

    def update_most_recent_state(self, new_state):
        self.states.append(new_state)

        if len(self.states) > self.larnn_window_size:
            self.states.popleft()

    def get_most_recent_cell(self):
        return self.states[-1]

    def get_past_cells_for_attention(self, use_positional_encoding=False):
        # Get the past states' inner cells
        past_cells = [state[1] for state in self.states]

        # Make that 1 tensor, not a list
        attention_values_tensor = torch.cat(past_cells, 0)

        # TODO: use_positional_encoding

        # IDEA: dropout on attention_tensor maybe?
        return attention_values_tensor


class LARNNCell(nn.Module):
    # Note: This LARNNCell class is inspired from the LSTM implementation here:
    # https://github.com/pytorch/benchmark/tree/master/benchmarks/lstm_variants
    # It has been cleaned and adapted here to the LARNN to have its multi-head
    # attention mechanism.

    def __init__(self, input_size, hidden_size, attention_heads, larnn_window_size,
                 larnn_mode='residual', use_positional_encoding=True, dropout=0.0):
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
        self.attention_heads = attention_heads
        self.dropout = dropout
        assert ((hidden_size / attention_heads) % 1 == 0.0), "'hidden_size' must be divisible by 'attention_heads'."

        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.bn_preact = torch.nn.BatchNorm1d(4 * hidden_size)

        self.c2v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.ih2q = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        if larnn_mode == 'residual':
            # Attention will be added to Wx and Wh as `Wx*x + Wh*h + Wa*a + bias1`.
            self.a2c = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        elif larnn_mode == 'layer':
            # Attention will be post-processed like `Wa*(concat(x, h, a)) + bias2`
            self.a2c = nn.Linear(3 * hidden_size, 4 * hidden_size, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, state):
        x = input
        previous_state = state.get_most_recent_cell()
        h, c = previous_state

        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        preact = self.linear_attention(x, h, state)

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
        h_t = torch.mul(o_t, c_t.tanh())

        if self.training and self.dropout > 0.0:
            F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)

        # Reshape for compatibility
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)

        # Update Linear Attention's Values
        current_state = (h_t, c_t)
        state.update_most_recent_state(current_state)

        return h_t, state

    def linear_attention(self, x, h, state):
        prev_cells = state.get_past_cells_for_attention(
            use_positional_encoding=self.use_positional_encoding)  # shape (larnn_window_size, batch_size, hidden_size)

        # `V = K = Wk*(a "larnn_window_size" number of most recent cells)`
        values = V = K = self.c2v(prev_cells)  # shape (larnn_window_size, batch_size, hidden_size)

        # `Q = Wxh*concat(x, h) + bxh`
        ih = torch.cat([x, h], -1)  # Concat on features
        query = Q = self.ih2q(ih)  #   # shape (batch_size, hidden_size)

        # `a(K, Q, V) = MultiHeadSoftmax(Q*K'/sqrt(dk))*V` like in Attention Is All You Need (AIAYN).
        # attention = self.multi_head_attention(query, values)  # TODO: multi-head
        attention = self.compute_attention_heads(query, values)

        if self.larnn_mode == 'residual':
            # Attention will be added to Wx and Wh as `Wx*x + Wh*h + Wa*a + b`.
            Wx_Wh_Wa_b = self.i2h(x) + self.h2h(h) + self.a2c(attention)
            preact = Wx_Wh_Wa_b
        elif self.larnn_mode == 'layer':
            # Attention will be post-processed like `Wa*(concat(x, h, a)) + b`
            Wxha_b = torch.cat([x, h, attention], -1)  # Concat on features
            preact = self.a2c(Wx_Wh_b_a)
        else:
            raise ValueError("'larnn_mode' must take the string value 'residual' or 'layer', not {}".format(self.larnn_mode))

        preac = self.bn_preact(preact)
        return preact

    def multi_head_attention(self, query, values):

        # TODO: debugs

        # hidden_size == attention_heads x d_k
        d_k = int(round(self.hidden_size / self.attention_heads))

        values_headed = values.view(self.state.batch_size, self.attention_heads, -1, d_k)

        values_headed = self.compute_attention_heads(query, values_headed)

        values_headed = values_headed.transpose(1, 2).view(
            self.state.batch_size, -1, hidden_size)

        return values_headed

    def compute_attention_heads(self, query, values):
        d_k = values.size(-1)

        print("values:", values.size())  # (larnn_window_size, batch_size, hidden_size)
        print("query:", query.size())  # (batch_size, hidden_size)

        scores = torch.matmul(query, values.transpose(2, 1)) / math.sqrt(d_k)
        print("scores:", scores.size())  # (larnn_window_size, batch_size, batch_size)

        p_attention = F.softmax(scores, -1)
        print("p_attention:", p_attention.size())  # (larnn_window_size, batch_size, batch_size)

        attention_head_result = torch.matmul(p_attention, values)
        print("attention_head_result:", attention_head_result.size())  # (larnn_window_size, batch_size, hidden_size)

        return attention_head_result


if __name__ == '__main__':
    """Debug the model."""

    for use_positional_encoding in [False, True]:
        for is_stacked_residual in [False, True]:
            for larnn_mode in ['concat', 'residual', 'layer']:

                print(
                    "use_positional_encoding, is_stacked_residual, larnn_mode:",
                    use_positional_encoding, is_stacked_residual, larnn_mode)

                larnn = LARNN(
                    input_size, hidden_size, num_layers,
                    is_stacked_residual=is_stacked_residual,
                    larnn_mode=larnn_mode, dropout=0.0)

                input_shape = (
                    hyperparameters_sample['time_steps'],
                    hyperparameters_sample['batch_size'],
                    hyperparameters_sample['input_size'])
                X_train = torch.autograd.Variable(
                    torch.rand(input_shape), requires_grad=True)
                _output, _hidden = larnn(X_train)

                print(_output.size())
                print(_hidden)
                print("")

                del larnn
                del _output
                del _hidden
