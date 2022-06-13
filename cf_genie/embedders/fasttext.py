from typing import List

from gensim.models import FastText

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.embedders.base import BaseEmbedder
from cf_genie.utils import Timer

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


class FastTextEmbedder(BaseEmbedder):
    def __init__(self, docs_to_train_embedder: List[List[str]]):
        super().__init__(docs_to_train_embedder)

        model_path = utils.get_model_path('fasttext.bin')
        try:
            model = FastText.load(model_path)
        except BaseException:
            log.info('Model not saved, building it from scratch')
            model = FastText(epochs=30, workers=utils.CORES)
            model.build_vocab(docs_to_train_embedder)

            with Timer(f'FastText training {model}', log=log):
                model.train(
                    docs_to_train_embedder,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
            model.save(model_path)

        self.model: FastText = model

    def embed(self, docs: List[str]):
        return self.model.wv.get_sentence_vector(docs)
