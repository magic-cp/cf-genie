"""
Long-short Term Memory (LSTM) model.
"""

from typing import Any, Dict, List, Optional

import keras
from hyperopt import hp
from keras import Sequential, callbacks, layers
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from itertools import takewhile

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.models.base import BaseSupervisedModel, CustomKerasClassifier, TrainingMethod
from cf_genie.utils import get_model_path


log = logger.get_logger(__name__)


class MLP(BaseSupervisedModel):
    TRAINING_METHOD = TrainingMethod.GRID_SEARCH_CV

    def init_model_object(self, **params) -> Sequential:

        def get_clf_model(num_hidden_layers_1: int,
                        num_hidden_layers_2: Optional[int],
                        num_hidden_layers_3: Optional[int],
                          meta: Dict[str,
                                     Any],
                          compile_kwargs: Dict[str,
                                               Any]) -> Sequential:
            model = Sequential(name='MLP-cf-genie')

            model.add(layers.Input(shape=(meta['n_features_in_'],)))

            model.add(layers.Normalization(axis=-1, name='normalization'))

            activation = 'relu'
            layer_sizes = takewhile(lambda x: x is not None and x <= meta['n_features_in_'], [num_hidden_layers_1, num_hidden_layers_2, num_hidden_layers_3])
            for i, layer_size in enumerate(layer_sizes):
                model.add(layers.Dense(layer_size, activation=activation, name=f'hidden-layer-{i}'))



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

            model.add(layers.Dense(n_output_units, name='output', activation=output_activation))

            model.compile(loss=loss, metrics=metrics, optimizer=compile_kwargs['optimizer'])

            # model.summary()
            return model

        early_stopping = callbacks.EarlyStopping(patience=2, monitor='loss')

        # csv_logger = callbacks.CSVLogger(self.model_name, append=True)

        clf = CustomKerasClassifier(
            model=get_clf_model,
            epochs=10000,
            verbose=0,
            optimizer='adam',
            optimizer__learning_rate=0.0005,
            callbacks=[early_stopping]
        )
        return clf

    @staticmethod
    def _get_search_space():
        return {param: hp.choice(param, choices)
                for param, choices in MLP._param_grid_for_grid_search().items()}

    @property
    def model(self) -> CustomKerasClassifier:
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
            'model__num_hidden_layers_1': [18, 28, 52, 78, 102],
            'model__num_hidden_layers_2': [None],
            'model__num_hidden_layers_3': [None],
        }
