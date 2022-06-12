from typing import Callable, List

import pandas as pd


class BaseEmbedder:
    """
    Represents and holds the information for a word embedder.

    Sub-classes should implement the `embed` method
    """

    def __init__(self, docs_to_train_embedder: List[List[str]]):
        """
        Initialize the embedder.

        :embeder_name: The name of the embedder.
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
