"""
Long-short Term Memory (LSTM) model.
"""

from typing import Any, Dict

import keras
from hyperopt import hp
from keras import Sequential, layers
from scikeras.wrappers import KerasClassifier
from tensorflow import keras

from cf_genie.models.base import BaseSupervisedModel
from cf_genie.utils import get_model_path


class KerasClassifierWithOneHotEncoding(KerasClassifier):
    @property
    def target_encoder(self):
        encoder = super().target_encoder
        # We have to set this value even for binary classification. Otherwise, the
        # target encoder won't use One hot encoding
        encoder.loss = 'categorical_crossentropy'

        return encoder


class LSTM(BaseSupervisedModel):
    @staticmethod
    def init_model_object(**params) -> Sequential:

        def get_clf_model(meta: Dict[str, Any], compile_kwargs: Dict[str, Any]) -> Sequential:
            model = Sequential(name='LSTM-cf-genie')

            model.add(
                layers.ZeroPadding1D(
                    padding=params['zero-padding-layer-padding'],
                    name='zero-padding-layer',
                    input_shape=(
                        meta['n_features_in_'],
                        1)))

            model.add(
                layers.Bidirectional(
                    layers.LSTM(
                        params['lstm-layer-1'],
                        name='lstm-layer-1',
                        return_sequences=params['lstm-layer-2'] is not None),
                    name='lstm-layer-1'))

            if params['lstm-layer-2'] is not None:
                model.add(layers.LSTM(params['lstm-layer-2'], name='lstm-layer-2', return_sequences=False))

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

            if params['dropout'] > 0:
                model.add(layers.Dropout(params['dropout'], name='dropout-layer'))

            model.add(layers.Dense(n_output_units, name='output', activation=output_activation))

            model.compile(loss=loss, metrics=metrics, optimizer=compile_kwargs['optimizer'])

            model.summary()
            return model

        clf = KerasClassifierWithOneHotEncoding(
            model=get_clf_model,
            epochs=40,
            batch_size=500,
            verbose=1,
            optimizer='adam',
            optimizer__learning_rate=0.001,
        )
        return clf

    @staticmethod
    def _get_search_space() -> object:
        return {
            'zero-padding-layer-padding': hp.choice('zero-padding-layer-padding', [1, 3, 5, 7]),
            'lstm-layer-1': hp.choice('lstm-layer-1', [16, 32, 64, 128, 256]),
            'lstm-layer-2': hp.choice('lstm-layer-2', [None, 8]),
            'dropout': hp.uniform('dropout', 0, 0.5),
        }

    @property
    def model(self) -> KerasClassifierWithOneHotEncoding:
        return self._model

    @staticmethod
    def get_fmin_kwargs():
        return {
            'max_evals': 80
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
