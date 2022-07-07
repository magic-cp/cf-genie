from typing import Callable, List, Type

from cf_genie.models.base import (BaseSupervisedModel, BaseUnSupervisedModel,
                                  TrainingMethod)
from cf_genie.models.complement_naive_bayes import ComplementNaiveBayes
from cf_genie.models.kmeans import KMeansClustering
from cf_genie.models.lstm import LSTM
from cf_genie.models.multinomial_naive_bayes import MultinomialNaiveBayes
from cf_genie.models.random_forest import RandomForest

SUPERVISED_MODELS: List[Type[BaseSupervisedModel]] = [
    # MultinomialNaiveBayes,
    # ComplementNaiveBayes,
    # RandomForest,
    LSTM,
]
UNSUPERVISED_MODELS: List[Type[BaseUnSupervisedModel]] = [KMeansClustering]
