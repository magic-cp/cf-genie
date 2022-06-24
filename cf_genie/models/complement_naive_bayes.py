"""
Complemeent naive bayes sklearn wrapper.

Complement naive bayes (CNB) is a veresion of MNB that is supposed to work well with imbalanced data
"""
from typing import List

from hyperopt import hp
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from cf_genie.models.base import BaseSupervisedModel


class ComplementNaiveBayes(BaseSupervisedModel):
    @classmethod
    def _get_search_space(cls):
        return {
            'alpha': hp.uniform('alpha', 0.0, 1.0),
        }

    @property
    def model(self) -> Pipeline:
        return self._model

    def init_model_object(self, params) -> object:
        return Pipeline([('scaler', MinMaxScaler()), ('estimator', ComplementNB(**params))])

    def predict(self, doc) -> List[str]:
        return self.model.predict(doc)
