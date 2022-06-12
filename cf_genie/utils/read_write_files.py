import os
from pathlib import Path

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))
DATASET_PATH = os.path.join(PROJECT_DIR, 'dataset')
PLOTS_PATH = os.path.join(PROJECT_DIR, 'plots')
TEMP_PATH = os.path.join(PROJECT_DIR, 'temp')

Path(DATASET_PATH).mkdir(parents=True, exist_ok=True)
Path(PLOTS_PATH).mkdir(parents=True, exist_ok=True)
Path(TEMP_PATH).mkdir(parents=True, exist_ok=True)


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


@_absolute_file_path(DATASET_PATH)
def _write_dataset(file_name: str, dataframe: pd.DataFrame):
    dataframe.to_csv(file_name, index=False)


def read_raw_dataset() -> pd.DataFrame:
    return _read_dataset('raw_cf_problems.csv')


def read_cleaned_dataset() -> pd.DataFrame:
    df = _read_dataset('cleaned_cf_problems.csv')
    df['preprocessed_statement'] = df['preprocessed_statement'].apply(lambda x: x.split(' '))
    return df


def write_cleaned_dataframe_to_csv(dataframe: pd.DataFrame):
    _write_dataset('cleaned_cf_problems.csv', dataframe)


@_absolute_file_path(PLOTS_PATH)
def write_plot(file_name, plt):
    dir_name_of_file = os.path.join(PROJECT_DIR, os.path.dirname(file_name))
    Path(dir_name_of_file).mkdir(parents=True, exist_ok=True)
    plt.savefig(file_name)
