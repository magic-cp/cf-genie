from typing import List


class BaseModel:
    """
    Represents and holds the information for a model.
    """

    def __init__(self, version=''):
        """
        Initialize the model.

        Subclasses has to call this method at the end of their __init__
        """
        self._model_name = type(self).__name__
        if version:
            self._model_name += '-' + version.replace(' ', '_')
        self.train()

    @property
    def model_name(self) -> str:
        return self._model_name

    def train(self) -> None:
        raise NotImplementedError("Subclasses of BaseMel should implement `train`")


class BaseSupervisedModel(BaseModel):
    """
    Represents and holds the information for a word embedder.

    Sub-classes should implement the `embed` method
    """

    def __init__(self, docs_to_train_model: List[List[float]], labels: List[str], version: str = ''):
        """
        Initialize the embedder.

        :docs_to_train_model: paragraphs to train the embedder on.
        """
        self._docs_to_train_models = docs_to_train_model
        self._labels = labels
        super().__init__(version)

    def training_score(self) -> float:
        raise NotImplementedError("Subclasses of BaseMel should implement `training_score`")

    def test_score(self, X, y) -> float:
        raise NotImplementedError("Subclasses of BaseMel should implement `test_score`")


class BaseUnSupervisedModel(BaseModel):
    """
    Represents and holds the information for a word embedder.

    Sub-classes should implement the `train` method
    """

    def __init__(self, docs_to_train_model: List[List[float]], version: str = ''):
        """
        Initialize the embedder.

        :docs_to_train_model: paragraphs to train the embedder on.
        """
        self._docs_to_train_models = docs_to_train_model
        super().__init__(version)
