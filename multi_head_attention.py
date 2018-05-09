
"""The Annotated Transformer Netowork's Multi-Head Attention.

For a walktrough of this code to gain intuition, see:
https://github.com/guillaume-chevalier/Linear-Attention-Recurrent-Neural-Network/blob/master/AnnotatedMultiHeadAttention.ipynb
"""

# The code in this file is all adapted from:
#     https://github.com/harvardnlp/annotated-transformer
#     MIT License, Copyright (c) 2018 Alexander Rush

# The edits are sublicensed as:
#     https://github.com/guillaume-chevalier/Linear-Attention-Recurrent-Neural-Network
#     MIT License, Copyright (c) 2018 Guillaume Chevalier
# Here, some things such as Attention Masks were removed.
# Also, there is no longer an extra linear layer on the query (so now there are
# only 3 linear clones, not 4).


import copy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def attention(query, key, value, dropout=None):
    # This function is adapted from:
    #     https://github.com/harvardnlp/annotated-transformer
    #     MIT License, Copyright (c) 2018 Alexander Rush
    "Compute 'Scaled Dot Product Attention'"
    # batch_size = 64
    # key_values_sequence_length = 10
    # query_sequence_length = 1
    # hidden_size = 32
    # attention_heads = 8
    d_k = query.size(-1)
    # print("    key 1:", key.size())  # key 1: torch.Size([64, 8, 10, 4])
    key = key.transpose(-2, -1)
    # print("    key 2:", key.size())  # key 2: torch.Size([64, 8, 4, 10])
    # print("    query:", query.size())  # query: torch.Size([64, 8, 1, 4])
    scores = torch.matmul(query, key) / math.sqrt(d_k)
    # print("    scores:", scores.size())  # scores: torch.Size([64, 8, 1, 10])
    p_attn = F.softmax(scores, dim = -1)
    # print("    p_attn:", p_attn.size())  # p_attn: torch.Size([64, 8, 1, 10])
    if dropout is not None:
        p_attn = dropout(p_attn)
    attention_result = torch.matmul(p_attn, value)
    # print("    attention_result:", attention_result.size())  # attention_result: torch.Size([64, 8, 1, 4])
    return attention_result, p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, input_size, hidden_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert hidden_size % h == 0
        # We assume d_v always equals d_k
        self.d_k = hidden_size // h
        self.h = h
        self.key_linear = nn.Linear(input_size, hidden_size)
        self.value_linear = nn.Linear(input_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Figure 2"
        # batch_size = 64
        # key_values_sequence_length = 10
        # query_sequence_length = 1
        # hidden_size = 32
        # attention_heads = 8
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from hidden_size => h x d_k
        # print("query, key, value 1:", query.size(), key.size(), value.size())  # query, key, value 1: torch.Size([64, 1, 32]) torch.Size([64, 10, 32]) torch.Size([64, 10, 32])
        key = self.key_linear(key)
        value = self.value_linear(value)
        query, key, value = [
            x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for x in (query, key, value)]
        # print("query, key, value 2:", query.size(), key.size(), value.size())  # query, key, value 2: torch.Size([64, 8, 1, 4]) torch.Size([64, 8, 10, 4]) torch.Size([64, 8, 10, 4])

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, self.dropout)
        # print("x 1:", x.size())  # x 1: torch.Size([64, 8, 1, 4])

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        # print("x 2:", x.size())  # x 2: torch.Size([64, 1, 32])

        x = self.output_linear(x)
        # print("x 3:", x.size())  # x 3: torch.Size([64, 1, 32])
        return x


class PositionalEncoding(nn.Module):
    # This class is adapted from:
    #     https://github.com/harvardnlp/annotated-transformer
    #     MIT License, Copyright (c) 2018 Alexander Rush
    # Is sublicensed:
    #     https://github.com/guillaume-chevalier/Linear-Attention-Recurrent-Neural-Network
    #     MIT License, Copyright (c) 2018 Guillaume Chevalier
    "Implement the edited PE function, depends on sequence length rather than input dimensionnality."

    def __init__(self, batch_size, max_sequence_length, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log_2 space ceiled to sequence_length.
        b = math.ceil(math.log(max_sequence_length * 4, 2))
        a = int(2**b / 4)  # Up to a quarter of a sine wave
        x1 = np.array([[math.cos(0.5**i*x*2*math.pi) for x in range(max_sequence_length, 0, -1)] for i in range(1, b+1)])
        x2 = np.array([[math.sin(0.5**i*x*2*math.pi) for x in range(max_sequence_length, 0, -1)] for i in range(2, b+2)])
        x = np.concatenate([x1, x2], axis=0)
        # print("x.shape():", x.shape)
        x = np.expand_dims(x, 0).repeat(repeats=batch_size, axis=0)
        # print("x.shape():", x.shape)

        # Register it into PyTorch
        pe = torch.from_numpy(x).float()
        pe = pe.transpose(-1, -2)
        # print("pe.size():", pe.size())
        self.register_buffer('pe', pe)

        self.positional_features = pe.size(-1)

    @staticmethod
    def get_features_dimensionnality(max_sequence_length):
        b = math.ceil(math.log(max_sequence_length * 4, 2))
        count = len(range(1, b+1)) + len(range(2, b+2))
        return count

    def forward(self, x):
        pos = Variable(self.pe, requires_grad=False)
        # print(pos.size(), x.size())  # [batch_size, -1, sequence_length], [batch_size, sequence_length, hidden_size]
        pe = self.pe[:, -x.size(1):]  # limiting positional encoding to a poentially shorter sequence_length
        # print("teretretretr", pe.size(), x.size())
        x = torch.cat([x, pe], dim=-1)
        return self.dropout(x)
