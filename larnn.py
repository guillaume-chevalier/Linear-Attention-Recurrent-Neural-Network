
"""PyTorch implementation of the LARNN, by Guillaume Chevalier"""

import torch
from torch import nn


__author__ = "Guillaume Chevalier"
__license__ = "MIT License"
__copyright__ = "Copyright 2018, Guillaume Chevalier"


class LARNNModel(nn.Module):

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
        'hidden_units': 42 * hp.loguniform('hidden_units_mult', -0.6, 0.6),

        # Use batch normalisation at more places?
        'use_BN': True,
        # Number of layers, either stacked or residualy stacked:
        'nb_layers': hp.choice('nb_layers', [2, 3]),,
        # Use residual connections for the 2nd (stacked) layer?
        'is_stacked_residual': hp.choice('is_stacked_residual', [False, True]),
        # How the new attention is placed in the LSTM
        'larnn_mode': hp.choice('attention_type', [
            'concat',  # Attention will be concatenated to x and h.
            'residual',  # Attention will be added to x and h.
            'layer'  # Attention will be computed from a layer with x and h.
        ]),
        # Wheter or not to use Positional Encoding similar to the one used in https://arxiv.org/abs/1706.03762
        'use_positional_encoding': hp.choice('use_positional_encoding', [False, True]),

        # Dataset-dependant values to be filled:
        'time_steps': None,
        'input_size': None
    }

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def forward(self, input, state=None):
        pass

class LARNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 is_stacked_residual=False, mode='residual', dropout=0.0):
        """A LARNN Cell on which it's possible to loop as an LSTM Cell.

        Args:
            input_size: number of features in the input `x`
            hidden_size: number of features in the inner LSTM's state
            num_layers: number of stacked layers in depth
            is_stacked_residual: wheter or not the stacked LARNN layers are
                added to the value of the first LARNN layer (and using a final
                batch norm).
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
            # If not using Truncated BPTT, init a new inner cell every time:
            state = LARNNCellState()
        pass

class LARNNCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                 mode='residual'):
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

    def forward(self, input, state):
        pass

    def _init_new_state(self):
        pass


class LARNNCellState(nn.Module):
    pass


if __name__ == '__main__':
    """Debug the model."""

    for use_positional_encoding in [False, True]:
        for is_stacked_residual in [False, True]:
            for larnn_mode in ['concat', 'residual', 'layer']:

                print(
                    "use_positional_encoding, is_stacked_residual, larnn_mode:",
                    use_positional_encoding, is_stacked_residual, larnn_mode)

                hyperparameters_sample = {
                    'learning_rate': 0.001,
                    'l2_weight_reg': 0.005,
                    'training_epochs': 80,
                    'dropout_drop_proba': 0.3,
                    'batch_size': 64,
                    'hidden_units': 42,
                    'use_BN': True,
                    'nb_layers': 2,
                    'is_stacked_residual': is_stacked_residual,
                    'larnn_mode': larnn_mode
                    'use_positional_encoding': use_positional_encoding,
                    # dataset-dependant values:
                    'time_steps': 5,
                    'input_size': 9
                }
                larnn = LARNNModel(hyperparameters=hyperparameters_sample)

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
