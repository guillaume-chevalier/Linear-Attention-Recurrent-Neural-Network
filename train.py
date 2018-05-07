
"""Train a model with Hyperopt, or retrain the best model in the main here."""

import os
import sys
import traceback
import uuid

from hyperopt import hp, STATUS_OK, STATUS_FAIL
import torch
from torch import nn

from json_utils import load_best_hyperspace, save_json_result, print_json
from larnn import LARNN
from datasets import UCIHARDataset, OpportunityDataset


__author__ = "Guillaume Chevalier"
__license__ = "MIT License"
__copyright__ = {
    "Version 1": "Copyright 2017, Guillaume Chevalier",
    "Version 2": "Copyright 2017, Vooban Inc.",
    "Version 3": "Copyright 2018, Guillaume Chevalier"
}
__notice__ = """
    Version 1, May 27 2017 - Jul 11 2017:
        Guillaume Chevalier
        Creation of the first version of file for the creation of a custom CIFAR-10 & CIFAR-100 CNN.
        See: https://github.com/guillaume-chevalier/Hyperopt-Keras-CNN-CIFAR-100/commit/7c2f8d5cadbfe96fb3f3572d07143f8ddbaa18d4#diff-06f0ae61dbe721276333a254a24a044b

    Version 2, Jul 19 2017 - Jul 25 2017:
        Guillaume Chevalier (On behalf of Vooban Inc.)
        Adapted the file for better training and visualizations.
        See: https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/commit/66c6492afa524139ba8153a8c7495cd177b08bf2#diff-6c53f5c58afef9e1fee290c207656b5e

    Version 3, May 6 2018 - May 11 2018:
        Guillaume Chevalier
        Adapted the file for the creation of its Linear Attention Recurrent Neural Network (LARNN).
        See: https://github.com/guillaume-chevalier/Linear-Attention-Recurrent-Neural-Network
"""


def optimize_model(hyperparameters, dataset, training_function):
    """Build a LARNN and train it on given dataset."""

    try:
        model, model_name, result = training_function(hyperparameters, dataset)

        # Save training results to disks with unique filenames
        save_json_result(model_name, dataset.NAME, result)

        # K.clear_session()
        del model

        return result

    except Exception as err:
        try:
            del model
        except:
            pass

        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
        return {
            'status': STATUS_FAIL,
            'err': err_str,
            'traceback': traceback_str
        }

    print("\n\n")


def train_for_uci_har(hyperparameters, dataset):
    """Build the deep CNN model and train it."""

    # Filling missing values of hyperparameters for what regards the dataset:
    hyperparameters['time_steps'] = 128
    hyperparameters['input_size'] = 9

    print("LARNN with hyperparameters:")
    print_json(hyperparameters)
    model = Model(hyperparameters)

    # Train net:
    # K.set_learning_phase(1)
    history = model.fit(
        [dataset.x_train],
        [dataset.y_train],
        batch_size=int(hyperparameters['batch_size']),
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
        validation_data=([dataset.x_test], [dataset.y_test])
    ).history

    # Test net:
    # K.set_learning_phase(0)
    score = model.evaluate([x_test], [y_test, y_test_coarse], verbose=0)
    max_acc = max(history['val_fine_outputs_acc'])

    model_name = "model_{}_{}".format(str(max_acc), str(uuid.uuid4())[:5])
    print("Model name: {}".format(model_name))
    print(history.keys())
    print(history)
    print(score)
    result = {
        # We plug "-val_fine_outputs_acc" as a
        # minimizing metric named 'loss' by Hyperopt.
        'loss': -max_acc,
        'real_loss': score[0],
        # Fine stats:
        'best_loss': min(history['val_fine_outputs_loss']),
        'best_accuracy': max(history['val_fine_outputs_acc']),
        'end_loss': score[1],
        'end_accuracy': score[3],
        # Misc:
        'model_name': model_name,
        'space': hyperparameters,
        'history': history,
        'status': STATUS_OK
    }
    print("RESULT:")
    print_json(result)

    return model, model_name, result


def train_for_opportunity(hyperparameters, dataset):
    return train_for_uci_har(hyperparameters, dataset)

dataset_name_to_class = {
    'UCIHAR': UCIHARDataset,
    'Opportunity': OpportunityDataset
}
dataset_name_to_training_function = {
    'UCIHAR': train_for_uci_har,
    'Opportunity': train_for_opportunity
}

def get_optimizer(dataset_name):
    _dataset = dataset_name_to_class[dataset_name]()
    _training_func = dataset_name_to_training_function[dataset_name]

    # Returns a callable for Hyperopt Optimization (for `fmin`):
    return lambda hyperparameters: (
        optimize_model(hyperparameters, _dataset, _training_func)
    )


class Model(nn.Module):
    HYPERPARAMETERS_SPACE = {
        # This loguniform scale will multiply the learning rate, so as to make
        # it vary exponentially, in a multiplicative fashion rather than in
        # a linear fashion, to handle his exponentialy varying nature:
        'learning_rate': 0.001 * hp.loguniform('learning_rate_mult', -0.5, 0.5),
        # L2 weight decay:
        'l2_weight_reg': 0.005 * hp.loguniform('l2_weight_reg_mult', -1.3, 1.3),
        # Number of loops on the whole train dataset
        'training_epochs': 80,
        # Number of examples fed per training step
        'batch_size': 64,

        # The dropout on the hidden unit on top of each LARNN cells
        'dropout_drop_proba': hp.uniform('dropout_drop_proba', 0.1, 0.5),
        # Let's multiply the "default" number of hidden units:
        'hidden_size': 42 * hp.loguniform('hidden_size_mult', -0.6, 0.6),
        # Use batch normalisation at more places?
        'use_BN': True,
        # Number of layers, either stacked or residualy stacked:
        'num_layers': hp.choice('num_layers', [2, 3]),
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

        self.larnn = LARNN(
            input_size=self.hyperparameters['input_size'],
            hidden_size=self.hyperparameters['hidden_size'],
            num_layers=self.hyperparameters['num_layers'],
            is_stacked_residual=self.hyperparameters['is_stacked_residual'],
            larnn_mode=self.hyperparameters['larnn_mode'],
            use_positional_encoding=self.hyperparameters['use_positional_encoding'],
            dropout=self.hyperparameters['dropout_drop_proba'],
        )

    def forward(self, input, state=None):
        return self.larnn(input, state)


if __name__ == "__main__":
    """Take the best hyperparameters and train on them."""

    dataset_name = 'UCIHAR'
    space_best_model = load_best_hyperspace(dataset_name)

    if space_best_model is None:
        print("You haven't found good hyperparameters yet. Run `hyperopt_optimize.py` first.")
        sys.exit(1)

    # Train the model.
    model, model_name, result = optimize_model(dataset_name)(space_best_model)

    # Prints training results to disks with unique filenames
    print("Model Name:", model_name)
    print("Training results (only printed here, not saved):")
    print_json(result)
