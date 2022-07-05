from enum import Enum, auto
from typing import Type

import numpy as np

import cf_genie.logger as logger
from cf_genie.embedders import EMBEDDERS
from cf_genie.embedders.base import BaseEmbedder
from cf_genie.models import BaseSupervisedModel
from cf_genie.models.base import TrainingMethod
from cf_genie.utils import Timer

log = logger.get_logger(__name__)


def all_strategy(model_class: Type[BaseSupervisedModel], embedder_class: Type[BaseEmbedder], y: np.ndarray):
    with Timer(f'Training {model_class.__name__} on embedder {embedder_class.__name__}'):

        model = model_class(
            embedder_class.read_embedded_words,
            y,
            TrainingMethod.GRID_SEARCH_CV,
            label='with-' +
            embedder_class.__name__ +
            '-on-all-classes')

    return model


class RunStrategy(Enum):
    ALL = auto()
    ONE_VS_ALL = auto()
    UNDERSAMPLING = auto()


STRATEGY_FUNS = {
    RunStrategy.ALL: all_strategy
}


def run_model(model_class: Type[BaseSupervisedModel], y: np.ndarray, run_strategy: RunStrategy):
    for embedder in EMBEDDERS:
        with Timer(f'Training {model_class.__name__} on embedder {embedder.__name__}'):
            if RunStrategy.ALL == run_strategy:
                fun = all_strategy
            else:
                raise NotImplementedError(run_strategy.__str__())
            fun(model_class, embedder, y)
