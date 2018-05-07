# LARNN: the Linear Attention Recurrent Neural Network

A fixed-size, go-back-`k` recurrent attention module on an RNN so as to have linear short-term memory by the means of attention. The LARNN model can be easily used inside a loop on the cell state just like any other RNN. The cell state keeps the `k` last states for its multi-head attention mechanism.

The LARNN is derived from the Long Short-Term Memory (LSTM) cell. The LARNN introduces attention on the state's past values up to a certain range, limited by a time window `k` to keep the forward processing linear in time in terms sequence length (time steps).

Therefore, multi-head attention with positional encoding is used on the most recent past values of the inner state cell so as to enable a better mid-term memory, such that at each new time steps, the cell looks back at it's own previous cell state values with an attention query.


## Downloading the datasets

```
cd data
python3 download_datasets.py
cd ..
```

## Meta-optimize the LARNN

```
python3 hyperopt_optimize.py --dataset UCIHAR
python3 hyperopt_optimize.py --dataset Opportunity
```

## Retrain on best hyperparameters found by meta-optimization

```
python3 train.py --dataset UCIHAR
python3 train.py --dataset Opportunity
```

## Debug/print the dimensions of the LARNN

```
python3 larnn.py
```

## References

- https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
- https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs
- https://github.com/sussexwearlab/DeepConvLSTM
- https://github.com/guillaume-chevalier/Hyperopt-Keras-CNN-CIFAR-100
- https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100
- https://github.com/harvardnlp/annotated-transformer
- https://github.com/pytorch/examples
- https://github.com/pytorch/pytorch
- https://github.com/pytorch/benchmark
- https://github.com/hyperopt/hyperopt

## License

[MIT License](https://github.com/guillaume-chevalier/Linear-Attention-Recurrent-Neural-Network/blob/master/LICENSE)

Copyright (c) 2018 Guillaume Chevalier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
