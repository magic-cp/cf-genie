import tempfile
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
from hyperopt import STATUS_OK
from sklearn.metrics import f1_score, hamming_loss, make_scorer
from sklearn.model_selection import GridSearchCV, cross_validate

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.utils import Timer
from cf_genie.utils.exceptions import ModelTrainingException
from cf_genie.utils.read_write_files import (write_grid_search_cv_results,
                                             write_hyper_parameters)


class TrainingMethod(Enum):
    """
    Choices for possible training methods.
    """

    HYPEROPT = auto()
    """
    Use hyperopt to find the best parameters. This would require to have a MongoDB instance running just like described on the README file.
    """

    GRID_SEARCH_CV = auto()
    """
    Use scikit-learn GridSearchCV class to run grid-search on your classifier template.
    """

    DEFER_TO_MODEL = auto()
    """
    Defer to the sub-class implementation
    """


class BaseModel(logger.Loggable):
    """
    Represents and holds the information for a model.
    """

    def __init__(self, label: str = ''):
        """
        Initialize the model.

        Subclasses has to call this method at the end of their __init__
        """
        super().__init__()
        self._model_name = type(self).__name__
        if label:
            self._model_name += '-' + label.replace(' ', '_')
        self.train()

    @property
    def model_name(self) -> str:
        return self._model_name

    def train(self) -> None:
        raise NotImplementedError("Subclasses of BaseModel should implement `train`")


class BaseSupervisedModel(BaseModel):
    """
    Represents a supervised model that can be trained and tested.

    Subclasses should implement:

    - _get_search_space: A dictionary of hyper-parameters to search over.
    - init_model_object: A function that returns a model object. This function is called when the model is not stored.
    - train: A function that trains the model.
    """
    SCORERS = {
        'f1_micro': 'f1_micro',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted',
        'hamming_score': make_scorer(hamming_loss, greater_is_better=False),
    }
    TRAINING_METHOD = None

    def __init__(self,
                 X_getter: Callable[[], List[List[float]]],
                 y: List[str],
                 label: str = ''):
        """
        Initialize the embedder.

        :docs_to_train_model: paragraphs to train the embedder on.
        """
        self._X_getter = X_getter
        self._y = y
        if self.TRAINING_METHOD is None:
            raise ValueError(
                f'Training method not specified for {self.__class__.__name__}. Did you forget to set the TRAINING_METHOD class attribute?')
        super().__init__(label)

    # @staticmethod
    def init_model_object(self, **params) -> object:
        raise NotImplementedError("Subclasses of BaseSupervisedModel should implement `init_model_object`")

    @staticmethod
    def _get_search_space() -> object:
        raise NotImplementedError("Subclasses of BaseSupervisedModel should implement `_get_search_space`")

    @staticmethod
    def get_fmin_kwargs():
        return {
            'max_evals': 60,
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
        with Timer(f'Loading X from disk', log=log):
            X = X_getter()

        model = cls.init_model_object(**params)
        log.info('model memory address: %s', hex(id(model)))

        with Timer(f'Training {model_name} {model} model with params {params}', log=log):
            scores = cross_validate(
                model,
                X,
                y,
                cv=10,
                scoring=cls.SCORERS,
                # return_train_score=True,
                n_jobs=2, verbose=1)

        log.info('scores: %s', scores)
        results = {
            'loss': -scores['test_f1_micro'].mean(),
            'status': STATUS_OK,
        }
        scorers = ['test_' + scorer for scorer in scorers]
        results.update({key: scores[key].mean() for key in scores if key in scorers})
        log.debug('Results: %s', results)
        return results

    def _read_model_from_disk(self) -> Any:
        return utils.read_model_from_file(self.model_path)

    @property
    def model_path(self):
        return self.model_name + '.pkl'

    def _save_model_to_disk(self, model) -> Any:
        utils.write_model_to_file(self.model_path, model)

    def train(self):
        try:
            self.log.debug('Attempting to load %s from disk storage', self.model_path)
            model = self._read_model_from_disk()
            self.log.debug('Model %s found on disk', self.model_path)

        except FileNotFoundError:
            self.log.info('Model %s not stored', self.model_name)
            model = self._train_if_not_in_disk()
            self._save_model_to_disk(model)

        except Exception:
            raise ModelTrainingException(self.model_name)

        self._model = model

    def _train_if_not_in_disk(self):
        model_name = self.model_name

        if self.TRAINING_METHOD == TrainingMethod.HYPEROPT:
            self.log.info('Building %s model from scratch using hyper-parameterization', model_name)

            with Timer(f'{model_name} hyper-parameterization', log=self.log):
                hyperopt_info = utils.run_hyperopt(
                    partial(
                        self._objective_fn_for_hyperopt,
                        self._X_getter,
                        self._y,
                        model_name,
                        self.log),
                    self._get_search_space(),
                    mongo_exp_key=model_name,
                    # store_in_mongo=False,
                    kwrgs=self.get_fmin_kwargs())
                model = self.init_model_object(**hyperopt_info.best_params_evaluated_space)
                model.fit(self._X_getter(), self._y)
        elif self.TRAINING_METHOD == TrainingMethod.GRID_SEARCH_CV:
            model = self._grid_search_model()
        elif self.TRAINING_METHOD == TrainingMethod.DEFER_TO_MODEL:
            model = self.init_model_object()
            model.fit(self._X_getter(), self._y)
        else:
            raise NotImplementedError(f'Training method {self.TRAINING_METHOD} not implemented')
        return model

    def _grid_search_model(self):
        model_name = self.model_name
        self.log.info('Building %s model from scratch doing a grid-search', model_name)

        clf = GridSearchCV(self.init_model_object(), self.__class__._param_grid_for_grid_search(), cv=5,
                           n_jobs=1, verbose=4, scoring=self.SCORERS, return_train_score=True, refit='f1_micro')

        with Timer(f'{model_name} grid-search', log=self.log):
            clf.fit(self._X_getter(), self._y)

        write_hyper_parameters(model_name, clf.best_params_)
        write_grid_search_cv_results(model_name, clf.cv_results_)

        return clf.best_estimator_

    @staticmethod
    def _param_grid_for_grid_search():
        raise NotImplementedError('When using grid search, you should define _param_grid_for_grid_search')


class BaseUnSupervisedModel(BaseModel):
    """
    Represents and holds the information for a word embedder.

    Sub-classes should implement the `train` method
    """

    def __init__(self, X: List[List[float]], label: str = ''):
        """
        Initialize the embedder.

        :docs_to_train_model: paragraphs to train the embedder on.
        """
        self._X = X
        super().__init__(label)
