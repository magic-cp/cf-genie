from typing import List, Optional

from gensim.models import FastText

import cf_genie.utils as utils
from cf_genie.embedders.base import BaseEmbedder
from cf_genie.utils import Timer


class FastTextEmbedder(BaseEmbedder):
    def __init__(self, size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_path = utils.get_model_path(f'{self.embedder_name}.bin')
        try:
            model = FastText.load(model_path)
        except BaseException:
            self.log.info('Model not saved, building it from scratch')
            model = FastText(epochs=30, workers=utils.CORES, vector_size=size or 100)
            model.build_vocab(self.docs_to_train_embedder)

            with Timer(f'FastText training {model}', log=self.log):
                model.train(
                    self.docs_to_train_embedder,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
            model.save(model_path)

        self.model: FastText = model

    def embed(self, docs: List[str]):
        return self.model.wv.get_sentence_vector(docs)


class FastTextEmbedder30(FastTextEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(30, *args, **kwargs)


class FastTextEmbedder50(FastTextEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(50, *args, **kwargs)


class FastTextEmbedder100(FastTextEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(100, *args, **kwargs)


class FastTextEmbedder150(FastTextEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(150, *args, **kwargs)


class FastTextEmbedder200(FastTextEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(200, *args, **kwargs)
