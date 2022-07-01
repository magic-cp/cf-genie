"""
Long-short Term Memory (LSTM) model.
"""

from typing import Any, Callable, Dict, List

import keras
from hyperopt import hp
from keras import Sequential, layers
from keras.utils import np_utils
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

import cf_genie.logger as logger
from cf_genie.models.base import BaseSupervisedModel
from cf_genie.utils import get_model_path

log = logger.get_logger(__name__)


class LSTM(BaseSupervisedModel):
    @staticmethod
    def init_model_object(**params) -> Sequential:
        log.info('HP-PARAMS: %s', params)

        def get_clf_model(meta: Dict[str, Any], compile_kwargs: Dict[str, Any]) -> Sequential:
            model = Sequential(name='LSTM-cf-genie')

            model.add(
                layers.ZeroPadding1D(
                    padding=3,
                    name='zero-padding-layer',
                    input_shape=(
                        meta['n_features_in_'],
                        1)))

            model.add(layers.Bidirectional(layers.LSTM(16, name='lstm-layer', return_sequences=True)))

            model.add(layers.LSTM(50, name='lstm-layer-2', return_sequences=False))

            if meta['target_type_'] == 'multiclass':
                n_output_units = meta['n_classes_']
                output_activation = 'softmax'
                loss = 'categorical_crossentropy'
                metrics = ['categorical_accuracy']
            elif meta['target_type_'] == 'binary':
                n_output_units = 1
                output_activation = 'sigmoid'
                loss = 'binary_crossentropy'
                metrics = ['binary_accuracy']
            else:
                raise ValueError('Model does not support target type: ' + meta['target_type_'])

            model.add(layers.Dense(n_output_units, name='output', activation=output_activation))

            model.compile(loss=loss, metrics=metrics, optimizer=compile_kwargs['optimizer'])

            model.summary()
            return model

        clf = KerasClassifier(
            model=get_clf_model,
            epochs=50,
            batch_size=500,
            verbose=1,
            # We have to set this value even for binary classification. Otherwise, the target encoder won't use One hot encoding
            loss='categorical_crossentropy',
            optimizer='adam',
            optimizer__learning_rate=0.001,
        )
        return clf

    @staticmethod
    def _get_search_space() -> object:
        return {
            'zero_padding_layer_padding': hp.choice('bla', [0])
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
