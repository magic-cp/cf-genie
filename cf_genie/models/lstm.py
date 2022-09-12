"""
Long-short Term Memory (LSTM) model.
"""

from typing import Any, Dict, Optional

import keras
from hyperopt import hp
from keras import Sequential, backend, callbacks, layers
from scikeras.wrappers import KerasClassifier
from tensorflow import keras

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.models.base import BaseSupervisedModel, TrainingMethod
from cf_genie.utils import get_model_path


class KerasClassifierWithOneHotEncoding(KerasClassifier):
    def fit(self, *args, **kwargs):
        backend.clear_session()  # to avoid high memory usage
        super().fit(*args, *kwargs)


log = logger.get_logger(__name__)


class LSTM(BaseSupervisedModel):
    TRAINING_METHOD = TrainingMethod.GRID_SEARCH_CV

    def init_model_object(self, **params) -> Sequential:

        def get_clf_model(zero_padding_layer_padding: int,
                          lstm_layer_1_num_nodes: int,
                          lstm_layer_2_num_nodes: int,
                          dropout: float,
                          extra_hidden_layer_num_nodes: Optional[int],
                          meta: Dict[str,
                                     Any],
                          compile_kwargs: Dict[str,
                                               Any]) -> Sequential:
            model = Sequential(name='LSTM-cf-genie')

            model.add(layers.Normalization(axis=-1, name='normalization', input_shape=(
                        meta['n_features_in_'],
                        1)))

            model.add(
                layers.ZeroPadding1D(
                    padding=zero_padding_layer_padding,
                    name='zero-padding-layer'))

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
                loss = 'sparse_categorical_crossentropy'
                metrics = ['sparse_categorical_accuracy']
            elif meta['target_type_'] == 'binary':
                n_output_units = 1
                output_activation = 'sigmoid'
                loss = 'binary_crossentropy'
                metrics = ['binary_accuracy']
            else:
                raise ValueError('Model does not support target type: ' + meta['target_type_'])

            metrics.append('mean_squared_error')

            if dropout > 0:
                model.add(layers.Dropout(dropout, name='dropout-layer'))

            if extra_hidden_layer_num_nodes:
                model.add(layers.Dense(extra_hidden_layer_num_nodes, name='extra-hidden', activation='relu'))

            model.add(layers.Dense(n_output_units, name='output', activation=output_activation))

            model.compile(loss=loss, metrics=metrics, optimizer=compile_kwargs['optimizer'])

            # model.summary()
            return model

        early_stopping = callbacks.EarlyStopping(patience=2, monitor='loss')

        # csv_logger = callbacks.CSVLogger(self.model_name, append=True)

        clf = KerasClassifierWithOneHotEncoding(
            model=get_clf_model,
            epochs=100,
            verbose=0,
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
        utils.write_lstm_history(self.model_name, model.model_.history.history)
        self.log.info('HISTORY: %s', model.model_.history.history)

    @staticmethod
    def _param_grid_for_grid_search():
        return {
            'model__zero_padding_layer_padding': [1, 3],
            'model__lstm_layer_1_num_nodes': [16, 32],
            'model__lstm_layer_2_num_nodes': [None, 4],
            'model__extra_hidden_layer_num_nodes': [None, 8],
            'model__dropout': [0, 0.25, 0.5],
        }
