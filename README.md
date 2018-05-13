# LARNN: the Linear Attention Recurrent Neural Network

A fixed-size, go-back-`k` recurrent attention module on an RNN so as to have linear short-term memory by the means of attention. The LARNN model can be easily used inside a loop on the cell state just like any other RNN. The cell state keeps the `k` last states for its multi-head attention mechanism.

The LARNN is derived from the Long Short-Term Memory (LSTM) cell. The LARNN introduces attention on the state's past values up to a certain range, limited by a time window `k` to keep the forward processing linear in time in terms sequence length (time steps).

Therefore, multi-head attention with positional encoding is used on the most recent past values of the inner state cell so as to enable a better mid-term memory, such that at each new time steps, the cell looks back at it's own previous cell state values with an attention query.


## Downloading the dataset

```
cd data
python3 download_dataset.py
cd ..
```

## Meta-optimize the LARNN

This will launch a round of meta-optimisation which will save the results under a new `./results/` folder.

```
python3 hyperopt_optimize.py --dataset UCIHAR --device cuda
```

Two training rounds have been executed and renamed under the folders `./results_round_1/` and `./results_round_2/` for now.

## Visualize the results

You can visually inspect the effect of every hyperparameter on the accuracy by navigating at:

- https://github.com/guillaume-chevalier/Linear-Attention-Recurrent-Neural-Network/blob/master/AnalyzeTestHyperoptResults_round_1.ipynb
- https://github.com/guillaume-chevalier/Linear-Attention-Recurrent-Neural-Network/blob/master/AnalyzeTestHyperoptResults_round_2.ipynb

You could run one of those files on new results by simply changing the results folder in the `jupyter-notebook` such that your new folder is taken.

## Edit the hyperparameters space

Here are the hyperparameters and their respective value ranges, which have been explored:

```
HYPERPARAMETERS_SPACE = {
    ### Optimization parameters
    # This loguniform scale will multiply the learning rate, so as to make
    # it vary exponentially, in a multiplicative fashion rather than in
    # a linear fashion, to handle his exponentialy varying nature:
    'learning_rate': 0.005 * hp.loguniform('learning_rate_mult', -0.5, 0.5),
    # How many epochs before the learning_rate is multiplied by 0.75
    'decay_each_N_epoch': hp.quniform('decay_each_N_epoch', 3 - 0.499, 10 + 0.499, 1),
    # L2 weight decay:
    'l2_weight_reg': 0.005 * hp.loguniform('l2_weight_reg_mult', -1.3, 1.3),
    # Number of loops on the whole train dataset
    'training_epochs': 25,
    # Number of examples fed per training step
    'batch_size': 256,

    ### LSTM/RNN parameters
    # The dropout on the hidden unit on top of each LARNN cells
    'dropout_drop_proba': hp.uniform('dropout_drop_proba', 0.05, 0.5),
    # Let's multiply the "default" number of hidden units:
    'hidden_size': 64 * hp.loguniform('hidden_size_mult', -0.6, 0.6),
    # The number 'h' of attention heads: from 1 to 20 attention heads.
    'attention_heads': hp.quniform('attention_heads', 6 - 0.499, 36 + 0.499, 1),

    ### LARNN (Linear Attention RNN) parameters
    # How restricted is the attention back in time steps (across sequence)
    'larnn_window_size': hp.uniform('larnn_window_size', 10, 50),
    # How the new attention is placed in the LSTM
    'larnn_mode': hp.choice('larnn_mode', [
        'residual',  # Attention will be added to Wx and Wh as `Wx*x + Wh*h + Wa*a + b`.
        'layer'  # Attention will be post-processed like `Wa*(concat(x, h, a)) + bs`
        # Note:
        #     `a(K, Q, V) = MultiHeadSoftmax(Q*K'/sqrt(dk))*V` like in Attention Is All You Need (AIAYN).
        #     `Q = Wxh*concat(x, h) + bxh`
        #     `V = K = Wk*(a "larnn_window_size" number of most recent cells)`
    ]),
    # Wheter or not to use Positional Encoding similar to the one used in https://arxiv.org/abs/1706.03762
    'use_positional_encoding': hp.choice('use_positional_encoding', [False, True]),
    # Wheter or not to use BN(ELU(.)) in the Linear() layers of the keys and values in the multi-head attention.
    'activation_on_keys_and_values': hp.choice('activation_on_keys_and_values', [False, True]),

    # Number of layers, either stacked or residualy stacked:
    'num_layers': hp.choice('num_layers', [2, 3]),
    # Use residual connections for the 2nd (stacked) layer?
    'is_stacked_residual': hp.choice('is_stacked_residual', [False, True])
}
```

The best results were found with those hyperparameters, for a test accuracy of 91.924%:

```
{
    "activation_on_keys_and_values": true,
    "attention_heads": 27,
    "batch_size": 256,
    "decay_each_N_epoch": 26,
    "dropout_drop_proba": 0.08885391813337816,
    "hidden_size": 81,
    "is_stacked_residual": true,
    "l2_weight_reg": 0.0006495900377590891,
    "larnn_mode": "residual",
    "larnn_window_size": 38,
    "learning_rate": 0.006026504115228934,
    "num_layers": 3,
    "training_epochs": 100,
    "use_positional_encoding": false
}
```

This can be seen at the end of the file `./results_round_2/UCIHAR/model_0.9192399049881235_b4105.txt.json`.

## Retrain on best hyperparameters found by meta-optimization

You can re-train on the best hyperparameters found with this command:

```
python3 train.py --dataset UCIHAR --device cuda
```

Note: before being able to run this command, you will need to have `.json` files from training results under the path `./results/UCIHAR/`. Currently, the best results are found within `./results_round_2/UCIHAR/`, the folder could be renamed to make this command work.

## Debug the LARNN model

This command is practical if you want to edit the model and potentially print-debug its dimensions:

```
python3 larnn.py
```

## References

The current project contains code derived from those other projects:

- https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
- https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs
- https://github.com/guillaume-chevalier/Hyperopt-Keras-CNN-CIFAR-100
- https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100
- https://github.com/harvardnlp/annotated-transformer

More information on which pieces of code comes from where in the headers of each Python files. All of those references are licensed under permissive open-source licenses, such as the MIT License and the Apache 2.0 License.

## License

My project is freely available under the terms of the [MIT License](https://github.com/guillaume-chevalier/Linear-Attention-Recurrent-Neural-Network/blob/master/LICENSE).

Copyright (c) 2018 Guillaume Chevalier

## Connect with me

- https://ca.linkedin.com/in/chevalierg
- https://twitter.com/guillaume_che
- https://github.com/guillaume-chevalier/
