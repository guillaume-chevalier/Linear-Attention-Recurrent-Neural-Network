
"""Json utils to print, save and load training results."""

from bson import json_util
import json
import os


__author__ = "Guillaume Chevalier"
__license__ = "MIT License"
__copyright__ = {
    "Version 1": "Copyright 2017, Vooban Inc."
}
__notice__ = """
    Version 1, Jul 19 2017 - Jul 25 2017:
        Guillaume Chevalier (On behalf of Vooban Inc.)
        Created file for proper JSON saving/loading of trained models.
        https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/commit/66c6492afa524139ba8153a8c7495cd177b08bf2#diff-04d2c9518498da64bfa9db44c6211645
"""


RESULTS_DIR = "results/"


def print_json(result):
    """Pretty-print a jsonable structure (e.g.: result)."""
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))


def save_json_result(model_name, dataset_name, result):
    """Save json to a directory and a filename."""
    result_name = '{}.txt.json'.format(model_name)
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    for dir in [RESULTS_DIR, results_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    with open(os.path.join(results_dir, result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def load_json_result(best_result_name, dataset_name):
    """Load json from a path (directory + filename)."""
    result_path = os.path.join(RESULTS_DIR, dataset_name, best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
            # default=json_util.default,
            # separators=(',', ': ')
        )


def load_best_hyperspace(dataset_name):
    results_dir = os.path.join(RESULTS_DIR, dataset_name)
    results = [
        f for f in list(sorted(os.listdir(results_dir))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result(best_result_name)["space"]
