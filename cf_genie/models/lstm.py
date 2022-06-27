"""
Long-short Term Memory (LSTM) model.
"""

from typing import Any, Callable, Dict, List

from hyperopt import hp
from keras import Sequential, layers
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

import cf_genie.logger as logger
from cf_genie.models.base import BaseSupervisedModel

print()


class LSTM(BaseSupervisedModel):
    @staticmethod
    def init_model_object(input_size: int, **params) -> Sequential:
        model = Sequential()

        model.add(layers.LSTM(20, input_shape=(input_size)))

        model.summary()
        return model

    @staticmethod
    def _get_search_space() -> object:
        return {
            'bla': hp.uniform('bla', 0, 1)
        }

    @staticmethod
    def get_fmin_kwargs():
        return {
            'max_evals': 1
        }

    def predict(self, X) -> Any:
        raise NotImplementedError("Subclasses of BaseSupervisedModel should implement `predict`")

    @classmethod
    def _objective_fn_for_hyperopt(cls,
                                   X_getter: Callable[[],
                                                      List[List[float]]],
                                   y,
                                   model_name,
                                   log: logger.Logger,
                                   params: Dict[str,
                                                Any]) -> Callable:
        X = X_getter()
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        skf = StratifiedKFold(n_splits=2)

        # Neural network plays better with one-hot encoding of output labels.
        label_encoder = LabelEncoder().fit(y)
        dummy_y = np_utils.to_categorical(label_encoder.transform(y))

        model = Sequential()

        model.add(layers.LSTM(20, input_shape=(X.shape[1:])))
        model.add(layers.Dense(dummy_y.shape[1], activation='softmax'))
        model.summary()

        model.compile()

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train, epochs=10)

        pass
