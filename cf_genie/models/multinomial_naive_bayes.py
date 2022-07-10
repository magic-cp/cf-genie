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

from cf_genie.models.base import BaseSupervisedModel, TrainingMethod


class MultinomialNaiveBayes(BaseSupervisedModel):
    TRAINING_METHOD = TrainingMethod.GRID_SEARCH_CV

    @staticmethod
    def _get_search_space():
        return {param: hp.choice(param, choices)
                for param, choices in MultinomialNaiveBayes._param_grid_for_grid_search().items()}

    @property
    def model(self) -> Pipeline:
        return self._model

    # @staticmethod
    def init_model_object(self, **params) -> object:
        return Pipeline([('scaler', MinMaxScaler()), ('estimator', MultinomialNB())]).set_params(**params)

    def predict(self, doc) -> List[str]:
        return self.model.predict(doc)

    @staticmethod
    def get_fmin_kwargs():
        return {
            'max_evals': 15,
        }

    @staticmethod
    def _param_grid_for_grid_search():
        return {
            'estimator__alpha': [0.2, 0.5, 0.8, 1.0, 10, 100, 250, 500, 1000]
        }
