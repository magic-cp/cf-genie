"""
Multinomial naive bayes sklearn wrapper.

Multinomial naive bayes (MNB) is a classifier that generalises Naive Bayes Classifier for multinomial distributions i.e.
allow us to perform multi-class classifications
"""
from typing import List

from hyperopt import hp
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from cf_genie.models.base import BaseSupervisedModel


class MultinomialNaiveBayes(BaseSupervisedModel):
    @staticmethod
    def _get_search_space():
        return {
            'estimator__alpha': hp.choice('alpha', [0.2, 0.5, 0.8, 1.0, 10, 100, 250, 500, 1000]),
        }

    @property
    def model(self) -> Pipeline:
        return self._model

    @staticmethod
    def init_model_object(**params) -> object:
        return Pipeline([('scaler', MinMaxScaler()), ('estimator', MultinomialNB())]).set_params(**params)

    def predict(self, doc) -> List[str]:
        return self.model.predict(doc)

    @staticmethod
    def get_fmin_kwargs():
        return {
            'max_evals': 15,
        }
