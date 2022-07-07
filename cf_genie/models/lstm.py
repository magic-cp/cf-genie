"""
Long-short Term Memory (LSTM) model.
"""

from typing import Any, Dict, Optional

import keras
from hyperopt import hp
from keras import Sequential, layers, callbacks
from scikeras.wrappers import KerasClassifier
from tensorflow import keras

import cf_genie.logger as logger
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


log = logger.get_logger(__name__)


class LSTM(BaseSupervisedModel):
    # @staticmethod
    def init_model_object(self, **params) -> Sequential:

        def get_clf_model(zero_padding_layer_padding: int, lstm_layer_1_num_nodes: int, lstm_layer_2_num_nodes: int,
                          dropout: float, extra_hidden_layer_num_nodes: Optional[int], meta: Dict[str, Any], compile_kwargs: Dict[str, Any]) -> Sequential:
            model = Sequential(name='LSTM-cf-genie')

            model.add(
                layers.ZeroPadding1D(
                    padding=zero_padding_layer_padding,
                    name='zero-padding-layer',
                    input_shape=(
                        meta['n_features_in_'],
                        1)))

            model.add(
                layers.Bidirectional(
                    layers.LSTM(
                        lstm_layer_1_num_nodes,
                        name='lstm-layer-1',
                        return_sequences=lstm_layer_2_num_nodes is not None),
                    name='lstm-layer-1'))

            if lstm_layer_2_num_nodes is not None:
                model.add(layers.LSTM(lstm_layer_2_num_nodes, name='lstm-layer-2', return_sequences=False))

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

            if dropout > 0:
                model.add(layers.Dropout(dropout, name='dropout-layer'))

            if extra_hidden_layer_num_nodes:
                model.add(layers.Dense(extra_hidden_layer_num_nodes, name='extra-hidden', activation='relu'))

            model.add(layers.Dense(n_output_units, name='output', activation=output_activation))

            model.compile(loss=loss, metrics=metrics, optimizer=compile_kwargs['optimizer'])

            model.summary()
            return model

        early_stopping = callbacks.EarlyStopping(patience=2, monitor='loss')

        clf = KerasClassifierWithOneHotEncoding(
            model=get_clf_model,
            epochs=50,
            batch_size=750,
            verbose=1,
            optimizer='adam',
            optimizer__learning_rate=0.001,
            callbacks=[early_stopping]
        )
        return clf

    @staticmethod
    def _get_search_space():
        return {param: hp.choice(param, choices)
                for param, choices in LSTM._param_grid_for_grid_search().items()}

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

    @staticmethod
    def _param_grid_for_grid_search():
        return {
            'model__zero_padding_layer_padding': [1, 3, 5, 7],
            'model__lstm_layer_1_num_nodes': [16, 32, 64, 128, 256],
            'model__lstm_layer_2_num_nodes': [None, 8],
            'model__extra_hidden_layer_num_nodes': [None, 4, 8, 16, 32],
            'model__dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
