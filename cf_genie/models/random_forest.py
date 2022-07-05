from typing import Any

from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier

from cf_genie.models.base import BaseSupervisedModel


class RandomForest(BaseSupervisedModel):
    @staticmethod
    def init_model_object(**params) -> object:
        return RandomForestClassifier(**params)

    @staticmethod
    def _get_search_space():
        return {param: hp.choice(param, choices)
                for param, choices in RandomForest._param_grid_for_grid_search().items()}

    @property
    def model(self) -> RandomForestClassifier:
        return self._model

    @staticmethod
    def get_fmin_kwargs():
        return {
            'max_evals': 15,
        }

    def predict(self, X) -> Any:
        return self.model.predict(X)

    @staticmethod
    def _param_grid_for_grid_search():
        return {
            'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500]
        }
