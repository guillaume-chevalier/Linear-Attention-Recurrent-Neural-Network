
"""PyTorch implementation of the LARNN, by Guillaume Chevalier"""

import torch
from torch import nn


__author__ = "Guillaume Chevalier"
__license__ = "MIT License"
__copyright__ = "Copyright 2018, Guillaume Chevalier"


class StackedLARNN(nn.Module):

    HYPERPARAMETERS_SPACE = {
        # This loguniform scale will multiply the learning rate, so as to make
        # it vary exponentially, in a multiplicative fashion rather than in
        # a linear fashion, to handle his exponentialy varying nature:
        'learning_rate': 0.001 * hp.loguniform('learning_rate_mult', -0.5, 0.5),
        # L2 weight decay:
        'l2_weight_reg': 0.005 * hp.loguniform('l2_weight_reg_mult', -1.3, 1.3),
        # Number of loops on the whole train dataset
        'training_epochs': 80,
        # Uniform distribution in finding appropriate dropout values, FC layers
        'dropout_drop_proba': 0.3,
        'batch_size': 64,

        # Let's multiply the "default" number of hidden units:
        'hidden_units': 64 * hp.loguniform('hidden_units_mult', -0.6, 0.6),

        # Use batch normalisation at more places?
        'use_BN': True,
        # Number of layers, either stacked or residualy stacked:
        'nb_layers': hp.choice('nb_layers', [2, 3]),,
        # Use residual connections for the 2nd (stacked) layer?
        'is_residual': hp.choice('is_residual', [False, True]),
        # How the new attention is placed in the LSTM
        'larnn_mode': hp.choice('attention_type', [
            'concat',  # Attention will be concatenated to x and h.
            'residual',  # Attention will be added to x and h.
            'layer'  # Attention will be computed from a layer with x and h.
        ]),
        # Wheter or not to use Positional Encoding similar to the one used in https://arxiv.org/abs/1706.03762
        'use_positional_encoding': hp.choice('use_positional_encoding', [False, True]),
    }

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def forward(self, input, state=None):
        pass

    def _init_new_state(self):
        pass

class LARNN(nn.Module):

    def __init__(self, input_size, hidden_size, mode='residual', dropout=0.0):
        """A LARNN Cell on which it's possible to loop as an LSTM Cell.

        Args:
            input_size: number of features in the input `x`
            hidden_size: number of features in the inner LSTM's state
            mode='concat|residual|layer': how the attention is plugged into the
                inner LSTM's layers. Specifically:
                  - 'concat' will concatenate the result of the attention to
                    the `(W*x+W*h+b)` values as `concat(W*x+W*h+b, a)`
                    before the gating and will hence make the inner cell bigger.
                  - 'residual' will add the attention result to the `(Wx+Wh+b)`
                    as (W*x+W*h+W*a+b).
                  - 'layer' will create a new layer such as
                    doing `W*(concat(W*x+W*h+b, a))`.
            dropout: how much dropout is applied on the output
        """
        raise NotImplementedError()

    def forward(self, input, state=None):
        if state is None:
            state = self._init_new_state()
        pass

    def _init_new_state(self):
        pass

class LARNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, mode='residual'):
        """A LARNN Cell on which it's possible to loop as an LSTM Cell.

        Args:
            input_size: number of features in the input `x`
            hidden_size: number of features in the inner LSTM's state
            mode='concat|residual|layer': how the attention is plugged into the
                inner LSTM's layers. Specifically:
                  - 'concat' will concatenate the result of the attention to
                    the `(W*x+W*h+b)` values as `concat(W*x+W*h+b, a)`
                    before the gating and will hence make the inner cell bigger.
                  - 'residual' will add the attention result to the `(Wx+Wh+b)`
                    as (W*x+W*h+W*a+b).
                  - 'layer' will create a new layer such as
                    doing `W*(concat(W*x+W*h+b, a))`.
        """
        raise NotImplementedError()

    def forward(self, input, state=None):
        if state is None:
            state = self._init_new_state()
        pass

    def _init_new_state(self):
        pass


class LARNNCellState(nn.Module):
    pass
