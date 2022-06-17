from typing import Callable, List

import numpy as np
import pandas as pd

import cf_genie.logger as logger
import cf_genie.utils as utils

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


class BaseEmbedder:
    """
    Represents and holds the information for a word embedder.

    Sub-classes should implement the `embed` method
    """

    def __init__(self, docs_to_train_embedder: List[List[str]]):
        """
        Initialize the embedder.

        :docs_to_train_embedder: paragraphs to train the embedder on.
        """
        self._embedder_name = type(self).__name__  # hat trick to get class name. Works for subclasses too
        self._docs_to_train_embedder = docs_to_train_embedder

    @property
    def embedder_name(self) -> str:
        return self._embedder_name

    @property
    def docs_to_train_embedder(self) -> List[List[str]]:
        return self._docs_to_train_embedder

    def embed(self, docs: List[str]) -> List[float]:
        raise NotImplementedError("This is not implemented yet")

    @classmethod
    def read_embedded_words(cls) -> np.ndarray:
        try:
            return utils.read_numpy_array(cls.__name__)
        except FileNotFoundError:
            raise ValueError(f'{cls.__name__} has not been trained yet. Run the embed_datasets task to fix this error')

    @classmethod
    def write_embedded_words(cls, n) -> None:
        utils.write_numpy_array(cls.__name__, n)
