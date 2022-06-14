from typing import Callable, List

from cf_genie.models.base import BaseSupervisedModel, BaseUnSupervisedModel
from cf_genie.models.complement_naive_bayes import ComplementNaiveBayes
from cf_genie.models.multinomial_naive_bayes import MultinomialNaiveBayes

SUPERVISED_MODELS: List[Callable[[List[List[float]], List[str]], BaseSupervisedModel]] = [MultinomialNaiveBayes]
UNSUPERVISED_MODELS: List[Callable[[], BaseUnSupervisedModel]] = []
