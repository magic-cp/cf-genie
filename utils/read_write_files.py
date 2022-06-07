import os

import pandas as pd

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dataset')

def _absolute_file_path(func):
    """
    Decorator to abstract adding the DATASET_PATH to each file name
    """
    def wrapped(*args, **kwargs):
        file_name = args[0]
        rest_args = args[1:]
        return func(os.path.join(DATASET_PATH, file_name), *rest_args, **kwargs)
    return wrapped

@_absolute_file_path
def _read_dataset(file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name)

def read_raw_dataset() -> pd.DataFrame:
    return _read_dataset('raw_cf_problems.csv')
