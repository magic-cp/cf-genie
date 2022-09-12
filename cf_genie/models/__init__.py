from typing import List, Type

from cf_genie.models.base import (BaseSupervisedModel, BaseUnSupervisedModel)
from cf_genie.models.complement_naive_bayes import ComplementNaiveBayes
from cf_genie.models.kmeans import KMeansClustering
from cf_genie.models.logistic_regression import LogisticRegression
from cf_genie.models.lstm import LSTM
from cf_genie.models.mlp import MLP
from cf_genie.models.multinomial_naive_bayes import MultinomialNaiveBayes
from cf_genie.models.random_forest import RandomForest

SUPERVISED_MODELS: List[Type[BaseSupervisedModel]] = [
    MultinomialNaiveBayes,
    ComplementNaiveBayes,
    RandomForest,
    LogisticRegression,
    LSTM,
    MLP,
]
UNSUPERVISED_MODELS: List[Type[BaseUnSupervisedModel]] = [KMeansClustering]
