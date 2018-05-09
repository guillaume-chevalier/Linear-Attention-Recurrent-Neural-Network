
"""The Annotated Transformer Netowork's Multi-Head Attention."""

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

import torch
from torch import nn
import torch.nn.functional as F


def clones(module, N):
    # This function is adapted from:
    #     https://github.com/harvardnlp/annotated-transformer
    #     MIT License, Copyright (c) 2018 Alexander Rush
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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

    def __init__(self, h, hidden_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert hidden_size % h == 0
        # We assume d_v always equals d_k
        self.d_k = hidden_size // h
        self.h = h
        self.linears = clones(nn.Linear(hidden_size, hidden_size), 3)
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
        key, value = self.linears[0](key), self.linears[1](value)
        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # print("query, key, value 2:", query.size(), key.size(), value.size())  # query, key, value 2: torch.Size([64, 8, 1, 4]) torch.Size([64, 8, 10, 4]) torch.Size([64, 8, 10, 4])

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, self.dropout)
        # print("x 1:", x.size())  # x 1: torch.Size([64, 8, 1, 4])

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        # print("x 2:", x.size())  # x 2: torch.Size([64, 1, 32])

        x = self.linears[-1](x)
        # print("x 3:", x.size())  # x 3: torch.Size([64, 1, 32])
        return x
