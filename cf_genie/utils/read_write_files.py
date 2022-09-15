import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import cf_genie.logger as logger

log = logger.get_logger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))
DATASET_PATH = os.path.join(PROJECT_DIR, 'dataset')
PLOTS_PATH = os.path.join(PROJECT_DIR, 'plots')
TEMP_PATH = os.path.join(PROJECT_DIR, 'temp')
MODELS_PATH = os.path.join(PROJECT_DIR, 'models')
HYPER_PARAMETERS_PATH = os.path.join(PROJECT_DIR, 'hyper-parameters')
TRAIN_RESULTS_PATH = os.path.join(PROJECT_DIR, 'train-results')
GRID_SEARCH_CV_RESULTS = os.path.join(PROJECT_DIR, 'grid-search-cv-results')
LSTM_HISTORY = os.path.join(PROJECT_DIR, 'lstm-history')

PATHS_TO_ENSURE = [
    DATASET_PATH,
    PLOTS_PATH,
    TEMP_PATH,
    MODELS_PATH,
    HYPER_PARAMETERS_PATH,
    TRAIN_RESULTS_PATH,
    GRID_SEARCH_CV_RESULTS,
    LSTM_HISTORY]

for path in PATHS_TO_ENSURE:
    Path(path).mkdir(parents=True, exist_ok=True)


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
def _read_dataset_from_csv(file_name: str) -> pd.DataFrame:
    return pd.read_csv(file_name)


@_absolute_file_path(DATASET_PATH)
def _write_dataset_to_csv(file_name: str, dataframe: pd.DataFrame):
    dataframe.to_csv(file_name, index=False)


def read_raw_dataset() -> pd.DataFrame:
    return _read_dataset_from_csv('raw_cf_problems.csv')


def get_cleaned_dataset_file_name(name_suffix: str = '') -> str:
    file_name_suffix = '' if not name_suffix else f'_{name_suffix}'
    return f'cleaned_cf_problems{file_name_suffix}.csv'


def read_cleaned_dataset(name_suffix: str = '') -> pd.DataFrame:
    df = _read_dataset_from_csv(get_cleaned_dataset_file_name(name_suffix))
    df['preprocessed_statement'] = df['preprocessed_statement'].apply(lambda x: x.split(' '))
    return df


def write_cleaned_dataframe_to_csv(df: pd.DataFrame, name_suffix=''):
    # Converting as a string separated by ' '
    df['preprocessed_statement'] = df['preprocessed_statement'].progress_apply(lambda row: ' '.join(row))

    # Same for the tag groups
    df['tag_groups'] = df['tag_groups'].progress_apply(lambda row: ' '.join(row))

    _write_dataset_to_csv(get_cleaned_dataset_file_name(name_suffix), df)


@_absolute_file_path(DATASET_PATH)
def read_numpy_array(file_name: str) -> np.ndarray:
    if not file_name.endswith('npy'):
        file_name += '.npy'
    return np.load(file_name)


@_absolute_file_path(DATASET_PATH)
def write_numpy_array(file_name: str, n: np.ndarray) -> None:
    return np.save(file_name, n)


@_absolute_file_path(PLOTS_PATH)
def write_plot(file_name, plt):
    dir_name_of_file = os.path.join(PROJECT_DIR, os.path.dirname(file_name))
    Path(dir_name_of_file).mkdir(parents=True, exist_ok=True)
    plt.savefig(file_name)


def get_model_path(model_name):
    return os.path.join(MODELS_PATH, model_name)


@_absolute_file_path(HYPER_PARAMETERS_PATH)
def write_hyper_parameters(model_name, hyper_parameters):
    file_name = model_name + '.json'
    with open(file_name, 'w') as f:
        json.dump(hyper_parameters, f, indent=4)


@_absolute_file_path(LSTM_HISTORY)
def write_lstm_history(model_name, hyper_parameters):
    file_name = model_name + '.json'
    with open(file_name, 'w') as f:
        json.dump(hyper_parameters, f, indent=4)


@_absolute_file_path(GRID_SEARCH_CV_RESULTS)
def write_grid_search_cv_results(model_name, grid_search_cv):
    file_name = model_name + '.csv'
    pd.DataFrame(grid_search_cv).to_csv(file_name)


@_absolute_file_path(TRAIN_RESULTS_PATH)
def write_train_results(model_name, results):
    file_name = model_name + '.json'
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)


@_absolute_file_path(MODELS_PATH)
def write_model_to_file(file_name, model, write_fun=pickle.dump):
    log.debug(f'Storing model to file: {file_name}')
    with open(file_name, 'wb') as f:
        write_fun(model, f)


@_absolute_file_path(MODELS_PATH)
def read_model_from_file(file_name, read_fun=pickle.load):
    log.debug(f'Reading model from file: {file_name}')
    with open(file_name, 'rb') as f:
        return read_fun(f)
