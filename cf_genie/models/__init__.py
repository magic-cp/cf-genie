from typing import Callable, List, Type

from cf_genie.models.base import BaseSupervisedModel, BaseUnSupervisedModel
from cf_genie.models.complement_naive_bayes import ComplementNaiveBayes
from cf_genie.models.multinomial_naive_bayes import MultinomialNaiveBayes
from cf_genie.models.random_forest import RandomForest

SUPERVISED_MODELS: List[Type[BaseSupervisedModel]] = [
    MultinomialNaiveBayes,
    ComplementNaiveBayes,
    RandomForest
]
UNSUPERVISED_MODELS: List[Type[BaseUnSupervisedModel]] = []
