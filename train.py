
"""Train a model with Hyperopt, or retrain the best model in the main here."""

from json_utils import print_json
from larnn import LARNN

from hyperopt import STATUS_OK, STATUS_FAIL

import uuid
import traceback
import sys
import os


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


def build_and_train(hyperparameters, dataset):
    """Build the deep CNN model and train it."""

    print("Hyperspace:")
    print(hyperparameters)

    K.set_learning_phase(1)
    K.set_image_data_format('channels_last')
    model = LARNN(hyperparameters)

    # Train net:
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
    K.set_learning_phase(0)
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


if __name__ == "__main__":
    """Take the best hyperparameters and train on them."""

    space_best_model = load_best_hyperspace()

    if space_best_model is None:
        print("You haven't found good hyperparameters yet. Run `hyperopt_optimize.py` first.")
        sys.exit(1)

    # Train the model.
    model, model_name, result = build_and_train(hyperparameters, dataset)

    # Save training results to disks with unique filenames
    print("Model Name:", model_name)
    print("Training results:")
    print_json(result)
