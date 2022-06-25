from typing import Any

from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier

from cf_genie.models.base import BaseSupervisedModel


class RandomForest(BaseSupervisedModel):
    @staticmethod
    def init_model_object(**params) -> object:
        return RandomForestClassifier(**params)

    @staticmethod
    def _get_search_space() -> object:
        return {
            'n_estimators': hp.choice('n_estimators', [100, 150, 200, 250, 300, 350, 400, 450, 500]),
        }

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
