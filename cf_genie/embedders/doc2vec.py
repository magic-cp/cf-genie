import collections
import pickle
from typing import Callable, List

import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from hyperopt import STATUS_OK, hp
from tqdm import tqdm

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.embedders.base import BaseEmbedder
from cf_genie.utils import Timer

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)

SEARCH_SPACE = {
    'vector_size': hp.choice('vector_size', [50, 100, 200, 300]),
    'dm': hp.choice('dm', [0, 1]),
    'window': hp.choice('window', [4, 5]),
    'hs': hp.choice('hs', [0, 1]),
}


def objective(tagged_docs: List[TaggedDocument]):
    def fn(params):
        log.info('Building word2vec model with params: %s', params)
        model = Doc2Vec(**params, negative=5, min_count=2, sample=0, workers=utils.CORES, epochs=40)

        model.build_vocab(tagged_docs)

        with Timer(f'Word2Vec training {model}', log=log):
            model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

        ranks = []
        with Timer(f"Doc2Vec inferring {model}", log=log):
            for doc_id in tqdm(range(len(tagged_docs)), desc='Doc2Vec inferring'):
                log.debug('doc_id: %s', doc_id)
                inferred_vector = model.infer_vector(tagged_docs[doc_id].words)
                log.debug('Inferred vector for doc id #%s: %s', doc_id, inferred_vector)
                sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
                log.debug('Similarity scores for doc id #%s: %s', doc_id, sims)
                rank = [docid for docid, sim in sims].index(doc_id)
                ranks.append(rank)

        counter = collections.Counter(ranks)

        simple_score = 100 * counter.get(0) / sum(counter.values())
        log.info('Simple score for model %s: %s', model, simple_score)

        return {
            'loss': -simple_score,
            'status': STATUS_OK,
            'attachments': {
                'model': pickle.dumps(model),
            }
        }

    return fn


class Doc2VecEmbedder(BaseEmbedder):
    def __init__(self, docs_to_train_embedder: List[List[str]]):
        super().__init__(docs_to_train_embedder)

        tagged_docs = self._tagged_docs()

        hyperopt_info = utils.run_hyperopt(objective(tagged_docs), SEARCH_SPACE, mongo_exp_key=self.embedder_name)

        self.model: Doc2Vec = hyperopt_info.best_model
        self.hyperopt_info = hyperopt_info

    def _tagged_docs(self) -> List[str]:
        return [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(self.docs_to_train_embedder)]

    def embed(self, doc: List[str]) -> pd.Series:
        return self.model.infer_vector(doc)
