import numpy as np
from hyperopt import hp
from sklearn.linear_model import LogisticRegressionCV as Logit

from .base import BaseSupervisedModel, TrainingMethod


class LogisticRegression(BaseSupervisedModel):
    TRAINING_METHOD = TrainingMethod.DEFER_TO_MODEL

    @staticmethod
    def _get_search_space():
        return {param: hp.choice(param, choices)
                for param, choices in LogisticRegression._param_grid_for_grid_search().items()}

    @property
    def model(self) -> Logit:
        return self._model

    def init_model_object(self, **params) -> object:
        return Logit(multi_class='ovr', cv=10, n_jobs=-1, max_iter=100000)

    def predict(self, doc) -> np.ndarray:
        return self.model.predict(doc)

    @staticmethod
    def get_fmin_kwargs():
        return {
            'max_evals': 15,
        }

    @staticmethod
    def _param_grid_for_grid_search():
        return {
            '_not_doing_hp_really': [1]
        }
