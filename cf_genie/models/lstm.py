"""
Long-short Term Memory (LSTM) model.
"""

from typing import Any, Callable, Dict, List

from hyperopt import hp
from keras import Sequential, layers
from keras.utils import np_utils
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import keras

import cf_genie.logger as logger
from cf_genie.models.base import BaseSupervisedModel
from cf_genie.utils import get_model_path


def get_clf_model(meta: Dict[str, Any]) -> Sequential:
    model = Sequential(name='LSTM-cf-genie')

    model.add(layers.LSTM(30, name='lstm-layer', input_shape=(meta['n_features_in_'], 1)))

    assert meta['target_type_'] == 'multiclass', 'Only multiclass target is supported'

    n_output_units = meta['n_classes_']
    output_activation = 'softmax'

    model.add(layers.Dense(n_output_units, name='output', activation=output_activation))

    model.summary()
    return model

class LSTM(BaseSupervisedModel):
    @staticmethod
    def init_model_object(**params) -> Sequential:

        clf = KerasClassifier(
            model=get_clf_model,
            epochs=50,
            batch_size=500,
            verbose=1,
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy'],
            optimizer='sgd',
            optimizer__learning_rate=0.001,
            optimizer__momentum=0.9
            )
        return clf

    @staticmethod
    def _get_search_space() -> object:
        return {
            'bla': hp.uniform('bla', 0, 1)
        }

    @property
    def model(self) -> KerasClassifier:
        return self._model

    @staticmethod
    def get_fmin_kwargs():
        return {
            'max_evals': 1
        }

    def predict(self, X) -> Any:
        return self.model.predict(X)

    @property
    def model_path(self) -> str:
        return get_model_path(self.model_name + '.hdf5')

    def _read_model_from_disk(self) -> Any:
        try:
            new_reg_model = keras.models.load_model(self.model_path)
            reg_new = KerasClassifier(new_reg_model)
            reg_new.initialize(self._X_getter(), self._y)
            return reg_new
        except OSError:
            raise FileNotFoundError('Model file not found: ' + self.model_path)

    def _save_model_to_disk(self, model) -> Any:
        model.model_.save(self.model_path)


    # @classmethod
    # def _objective_fn_for_hyperopt(cls,
    #                                X_getter: Callable[[],
    #                                                   List[List[float]]],
    #                                y,
    #                                model_name,
    #                                log: logger.Logger,
    #                                params: Dict[str,
    #                                             Any]) -> Callable:
    #     X = X_getter()
    #     X = X.reshape((X.shape[0], 1, X.shape[1]))
    #     skf = StratifiedKFold(n_splits=2)

    #     # Neural network plays better with one-hot encoding of output labels.
    #     label_encoder = LabelEncoder().fit(y)
    #     dummy_y = np_utils.to_categorical(label_encoder.transform(y))

    #     model = Sequential()

    #     model.add(layers.LSTM(20, input_shape=(X.shape[1:])))
    #     model.add(layers.Dense(dummy_y.shape[1], activation='softmax'))
    #     model.summary()

    #     model.compile()

    #     for train_index, test_index in skf.split(X, y):
    #         X_train, X_test = X[train_index], X[test_index]
    #         y_train, y_test = y[train_index], y[test_index]

    #         model.fit(X_train, y_train, epochs=10)

    #     pass
