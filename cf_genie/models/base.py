import tempfile
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
from hyperopt import STATUS_OK
from sklearn.model_selection import cross_validate

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.utils import Timer


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


def _objective_fn_for_hyperopt(X_file, Y, model_name, model_fn: Callable[[object], object], log: logger.Logger, params):
    X = np.load(X_file)

    model = model_fn(**params)

    with Timer(f'Training {model_name} model with params {params}', log=log):
        scores = cross_validate(
            model,
            X,
            Y,
            cv=5,
            scoring=(
                'f1_micro',
                'f1_macro',
                'f1_weighted'),
            return_train_score=True,
            n_jobs=-1)

    return {
        'loss': -scores['test_f1_micro'],
        'status': STATUS_OK,
        **scores,
    }


class BaseSupervisedModel(BaseModel):
    """
    Represents a supervised model that can be trained and tested.

    Subclasses should implement:

    - SEARCH_SPACE: A dictionary of hyper-parameters to search over.
    - init_model_object: A function that returns a model object. This function is called when the model is not stored.
    - train: A function that trains the model.
    """

    def __init__(self,
                 X: List[List[float]],
                 y: List[str],
                 label: str = ''):
        """
        Initialize the embedder.

        :docs_to_train_model: paragraphs to train the embedder on.
        """
        self._X = X
        self._y = y
        super().__init__(label)

    def init_model_object(self, params) -> object:
        raise NotImplementedError("Subclasses of BaseSupervisedModel should implement `init_model_object`")

    @classmethod
    def _get_search_space(cls) -> object:
        raise NotImplementedError("Subclasses of BaseSupervisedModel should implement `_get_search_space`")

    def train(self):
        model_name = self.model_name
        model_path = utils.get_model_path(f'{model_name}.pkl')
        try:
            model = utils.read_model_from_file(model_path)
        except BaseException:
            self.log.debug('Model not stored. Building %s model from scratch using hyper-parameterization', model_name)
            with tempfile.NamedTemporaryFile() as tempf_X:
                with Timer(f'Storing train and test arrays in files for model {model_name}', log=self.log):
                    np.save(tempf_X, self._X)

                with Timer(f'{model_name} hyper-parameterization', log=self.log):
                    hyperopt_info = utils.run_hyperopt(
                        partial(
                            _objective_fn_for_hyperopt,
                            tempf_X.name,
                            self._y,
                            model_name,
                            self.init_model_object,
                            self.log),
                        self.__class__._get_search_space(),
                        mongo_exp_key=model_name,
                        fmin_kwrgs={
                            'max_evals': 60})
                    model = hyperopt_info.best_model
                    utils.write_model_to_file(model_path, model)

        self._model = model

    def predict(self, X) -> Any:
        raise NotImplementedError("Subclasses of BaseMel should implement `predict`")


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
