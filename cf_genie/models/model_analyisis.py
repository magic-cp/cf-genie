from typing import Type

import pandas as pd

from cf_genie import utils
from cf_genie.embedders import EMBEDDERS, BaseEmbedder
from cf_genie.models import BaseSupervisedModel
from cf_genie.models.base import TrainingMethod
from cf_genie.models.model_runner import (
    get_model_suffix_name_for_all_classes,
    get_model_suffix_name_for_tag_vs_rest, get_model_suffix_name_without_tag)


def get_result_path_for_all_classes(model: Type[BaseSupervisedModel]):
    return f'grid-search-cv-results/{model.model_name}.csv'


def get_pandas_df_for_all_classes(
    model,
    embedder,
    scores=[
        'f1_micro',
        'f1_macro',
        'f1_weighted',
        'hamming_score']) -> pd.DataFrame:
    score_columns = []
    for score in scores:
        score_columns.append('rank_test_' + score)
        score_columns.append('mean_test_' + score)
        score_columns.append('std_test_' + score)
        score_columns.append('mean_train_' + score)
        score_columns.append('std_train_' + score)
    df = pd.read_csv(
        get_result_path_for_all_classes(
            model),
        usecols=[
            'mean_fit_time',
            'std_fit_time',
            'mean_score_time',
            'std_score_time',
            'params'] +
        score_columns,
        skipinitialspace=True,
        index_col=False)
    df['embedder'] = embedder.__name__
    return df


def get_acc_pandas_df_for_model_all_classes(
    model_class: Type[BaseSupervisedModel],
    scores=[
        'f1_micro',
        'f1_macro',
        'f1_weighted',
        'hamming_score']) -> pd.DataFrame:
    dfs = []
    y = utils.read_cleaned_dataset()['most_occurrent_tag_group'].to_numpy()
    for embedder in EMBEDDERS:
        try:
            df = get_pandas_df_for_all_classes(
                model_class(
                    embedder.read_embedded_words,
                    y,
                    TrainingMethod.GRID_SEARCH_CV,
                    label=get_model_suffix_name_for_all_classes(embedder)),
                embedder, scores)
            dfs.append(df)
        except Exception as e:
            print(embedder + ' not trained to model_class ' + model_class)
            print(e)

    df = pd.concat(dfs, ignore_index=True)
    if df is not None:
        cols = [col for col in df.columns.tolist() if col not in ['params', 'embedder']]
        cols = ['params', 'embedder'] + cols
        df = df[cols]
        return df
    return None
