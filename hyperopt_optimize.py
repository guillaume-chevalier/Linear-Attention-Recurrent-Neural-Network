
"""Auto-optimizing a neural network with Hyperopt (TPE algorithm)."""

import argparse
import os
import pickle
import traceback

from hyperopt import tpe, fmin, Trials

from train import get_optimizer, Model
from json_utils import RESULTS_DIR


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
        https://github.com/guillaume-chevalier/Hyperopt-Keras-CNN-CIFAR-100/commit/7c2f8d5cadbfe96fb3f3572d07143f8ddbaa18d4#diff-06f0ae61dbe721276333a254a24a044b

    Version 2, Jul 19 2017 - Jul 25 2017:
        Guillaume Chevalier (On behalf of Vooban Inc.)
        Adapted the file for better training and visualizations.
        https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/commit/66c6492afa524139ba8153a8c7495cd177b08bf2#diff-f12a5ae628b31ec5c8c0e16f563c0650

    Version 3, May 6 2018 - May 11 2018:
        Guillaume Chevalier
        Adapted the file for the creation of its Linear Attention Recurrent Neural Network (LARNN).
        https://github.com/guillaume-chevalier/Linear-Attention-Recurrent-Neural-Network
"""


parser = argparse.ArgumentParser(
    description='Hyperopt meta-optimizer for the LARNN Model on sensors datasets.')
parser.add_argument(
    '--dataset', type=str, default='UCIHAR',
    help='Which dataset to use ("UCIHAR" or "Opportunity")')
args = parser.parse_args()


def run_a_trial(args):
    """Run one TPE meta optimisation step and save its results."""
    results_pickle_file = os.path.join(RESULTS_DIR, "{}.pkl".format(args.dataset))

    max_evals = nb_evals = 1
    print("Attempt to resume a past training if it exists:")
    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open(results_pickle_file, "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        get_optimizer(args.dataset),
        Model.HYPERPARAMETERS_SPACE,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open(results_pickle_file, "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")


if __name__ == "__main__":
    """Run the optimisation forever (and save results)."""

    print("We will train many models, one after the other. "
          "Note that hyperopt has support for cloud "
          "distributed training using MongoDB.\n")

    print("Results will be saved in the folder named `results/`. "
          "You can sort that alphabetically and take the greatest one. "
          "As you run the optimization, results are consinuously saved into "
          "a `results.pkl` file, too. Re-running optimize.py will resume "
          "the meta-optimization to spawn new models in the given "
          "hyperparameters space.\n")

    while True:
        print("OPTIMIZING NEW MODEL:")
        try:
            run_a_trial(args)
        except Exception as err:
            err_str = str(err)
            print(err_str)
            traceback_str = str(traceback.format_exc())
            print(traceback_str)
