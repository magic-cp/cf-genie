import os
from pathlib import Path

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))
DATASET_PATH = os.path.join(PROJECT_DIR, 'dataset')
PLOTS_PATH = os.path.join(PROJECT_DIR, 'plots')

Path(DATASET_PATH).mkdir(parents=True, exist_ok=True)
Path(PLOTS_PATH).mkdir(parents=True, exist_ok=True)


def _absolute_file_path(DIR_NAME):
    """
    Decorator to abstract adding the DATASET_PATH to each file name. Decorated functions have to have a file_name as the
    first argument
    """
    def dec(func):
        def wrapped(*args, **kwargs):
            file_name = args[0]
            rest_args = args[1:]
            return func(
                os.path.join(
                    DIR_NAME,
                    file_name),
                *rest_args,
                **kwargs)
        return wrapped
    return dec


@_absolute_file_path(DATASET_PATH)
def _read_dataset(file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name)


def read_raw_dataset() -> pd.DataFrame:
    return _read_dataset('raw_cf_problems.csv')


@_absolute_file_path(PLOTS_PATH)
def write_plot(file_name, plt):
    plt.savefig(file_name)
