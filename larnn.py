
"""PyTorch implementation of the LARNN, by Guillaume Chevalier."""

import torch
from torch import nn


__author__ = "Guillaume Chevalier"
__license__ = "MIT License"
__copyright__ = "Copyright 2018, Guillaume Chevalier"


class LARNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 is_stacked_residual=False, larnn_mode='residual',
                 use_positional_encoding=True, dropout=0.0):
        """A LARNN Cell on which it's possible to loop as an LSTM Cell.

        Args:
            input_size: number of features in the input `x`
            hidden_size: number of features in the inner LSTM's state
            num_layers: number of stacked layers in depth
            is_stacked_residual: wheter or not the stacked LARNN layers are
                added to the value of the first LARNN layer (and using a final
                batch norm).
            larnn_mode='concat|residual|layer': how the attention is plugged
                into the inner LSTM's layers. Specifically:
                  - 'concat' will concatenate the result of the attention to
                    the `(W*x+W*h+b)` values as `concat(W*x+W*h+b, a)`
                    before the gating and will hence make the inner cell bigger.
                  - 'residual' will add the attention result to the `(Wx+Wh+b)`
                    as (W*x+W*h+W*a+b).
                  - 'layer' will create a new layer such as
                    doing `W*(concat(W*x+W*h+b, a))`.
            use_positional_encoding: wheter or not to use geometric series
                of sines and cosines for positional encoding, re-generated
                at each attention step
            dropout: how much dropout is applied on the output
        """
        super().__init__()
        raise NotImplementedError()

    def forward(self, input, state=None):
        if state is None:
            # If not using Truncated BPTT, init a new inner cell every time:
            state = LARNNCellState()
        pass

class LARNNCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                 larnn_mode='residual', use_positional_encoding=True):
        """A LARNN Cell on which it's possible to loop as an LSTM Cell.

        Args:
            input_size: number of features in the input `x`
            hidden_size: number of features in the inner LSTM's state
            larnn_mode='concat|residual|layer': how the attention is plugged
                into the inner LSTM's layers. Specifically:
                  - 'concat' will concatenate the result of the attention to
                    the `(W*x+W*h+b)` values as `concat(W*x+W*h+b, a)`
                    before the gating and will hence make the inner cell bigger.
                  - 'residual' will add the attention result to the `(Wx+Wh+b)`
                    as (W*x+W*h+W*a+b).
                  - 'layer' will create a new layer such as
                    doing `W*(concat(W*x+W*h+b, a))`.
            use_positional_encoding: wheter or not to use geometric series
                of sines and cosines for positional encoding, re-generated
                at each attention step
        """
        super().__init__()
        raise NotImplementedError()

    def forward(self, input, state):
        pass

    def _init_new_state(self):
        pass


class LARNNCellState(nn.Module):
    # todo: dropout on attention?
    pass


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
