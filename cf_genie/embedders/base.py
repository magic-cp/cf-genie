from typing import List

import numpy as np
import pandas as pd

import cf_genie.logger as logger
import cf_genie.utils as utils


class BaseEmbedder(logger.Loggable):
    """
    Represents and holds the information for a word embedder.

    Sub-classes should implement the `embed` method
    """

    def __init__(self, docs_to_train_embedder: List[List[str]]):
        """
        Initialize the embedder.

        :docs_to_train_embedder: paragraphs to train the embedder on.
        """
        super().__init__()
        self._embedder_name = self.__class__.__qualname__  # hat trick to get class name. Works for subclasses too
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
