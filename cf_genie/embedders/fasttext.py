from typing import List, Optional

from gensim.models import FastText

import cf_genie.utils as utils
from cf_genie.embedders.base import BaseEmbedder
from cf_genie.utils import Timer


class FastTextEmbedder(BaseEmbedder):
    def __init__(self, docs_to_train_embedder: List[List[str]], vector_size: Optional[int] = None):
        super().__init__(docs_to_train_embedder)

        model_path = utils.get_model_path(f'fasttext.bin' if vector_size is None else f'fasttext-{vector_size}.bin')
        try:
            model = FastText.load(model_path)
        except BaseException:
            self.log.info('Model not saved, building it from scratch')
            model = FastText(epochs=30, workers=utils.CORES, vector_size=vector_size or 100)
            model.build_vocab(docs_to_train_embedder)

            with Timer(f'FastText training {model}', log=self.log):
                model.train(
                    docs_to_train_embedder,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
            model.save(model_path)

        self.model: FastText = model

    def embed(self, docs: List[str]):
        return self.model.wv.get_sentence_vector(docs)

class FastTextEmbedder30(FastTextEmbedder):
    def __init__(self, docs_to_train_embedder: List[List[str]]):
        super().__init__(docs_to_train_embedder, 30)

class FastTextEmbedder50(FastTextEmbedder):
    def __init__(self, docs_to_train_embedder: List[List[str]]):
        super().__init__(docs_to_train_embedder, 50)

class FastTextEmbedder100(FastTextEmbedder):
    def __init__(self, docs_to_train_embedder: List[List[str]]):
        super().__init__(docs_to_train_embedder, 100)
class FastTextEmbedder150(FastTextEmbedder):
    def __init__(self, docs_to_train_embedder: List[List[str]]):
        super().__init__(docs_to_train_embedder, 150)

class FastTextEmbedder200(FastTextEmbedder):
    def __init__(self, docs_to_train_embedder: List[List[str]]):
        super().__init__(docs_to_train_embedder, 200)
