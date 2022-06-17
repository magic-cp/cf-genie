from typing import Callable, List, Type

from cf_genie.models.base import BaseSupervisedModel, BaseUnSupervisedModel
from cf_genie.models.complement_naive_bayes import ComplementNaiveBayes
from cf_genie.models.multinomial_naive_bayes import MultinomialNaiveBayes

SUPERVISED_MODELS: List[Type[BaseSupervisedModel]] = [
    MultinomialNaiveBayes,
    MultinomialNaiveBayes
]
UNSUPERVISED_MODELS: List[Type[BaseUnSupervisedModel]] = []
