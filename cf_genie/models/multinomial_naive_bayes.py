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
    @classmethod
    def _get_search_space(cls):
        return {
            'alpha': hp.uniform('alpha', 0.0, 1.0),
        }

    @property
    def model(self) -> Pipeline:
        return self._model

    def init_model_object(self, params) -> object:
        return Pipeline([('scaler', MinMaxScaler()), ('estimator', MultinomialNB(**params))])

    def predict(self, doc) -> List[str]:
        return self.model.predict(doc)
