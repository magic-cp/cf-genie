from typing import List

import numpy as np
from sklearn.cluster import KMeans

from cf_genie.models.base import BaseUnSupervisedModel


class KMeansClustering(BaseUnSupervisedModel):
    """
    Wraper over scikit-learn KMeans class
    """

    def __init__(self, X: List[List[float]], k: int, label: str = ''):
        self.k = k
        super().__init__(X, label)

    def train(self):
        self._model: KMeans = KMeans(n_clusters=self.k).fit(self._X)

    def __str__(self) -> str:
        return self._model.__str__()

    def predict(self, X: np.ndarray):
        return self._model.predict(X)
