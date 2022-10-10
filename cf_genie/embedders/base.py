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

    def __init__(self, docs_to_train_embedder: List[List[str]], label: str = '', training_label: str = ''):
        """
        Initialize the embedder.

        :docs_to_train_embedder: paragraphs to train the embedder on.
        """
        super().__init__()
        label = '' if not label else f'-{label}'
        training_label = '' if not training_label else f'-{training_label}'

        self._embedder_name = self.__class__.__qualname__  # hat trick to get class name. Works for subclasses too
        self._docs_to_train_embedder = docs_to_train_embedder
        self.label = label
        self.training_label = training_label

    @property
    def embedder_name(self) -> str:
        return self._embedder_name + self.training_label

    @property
    def embedder_name_with_label(self) -> str:
        return self._embedder_name + self.label

    @property
    def embedder_name_no_label(self) -> str:
        return self._embedder_name

    @property
    def docs_to_train_embedder(self) -> List[List[str]]:
        return self._docs_to_train_embedder

    def embed(self, docs: List[str]) -> List[float]:
        raise NotImplementedError("This is not implemented yet")

    def read_embedded_words(self) -> np.ndarray:
        try:
            return utils.read_numpy_array(self.embedder_name_with_label)
        except FileNotFoundError:
            raise ValueError(
                f'{self.embedder_name} has not been trained yet. Run the embed_datasets task to fix this error')

    def write_embedded_words(self, n) -> None:
        utils.write_numpy_array(self.embedder_name_with_label, n)
