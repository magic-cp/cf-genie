"""
Tiny wrapper over TfidfVectorizer
"""
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

import cf_genie.utils as utils
from cf_genie.embedders.base import BaseEmbedder
from cf_genie.utils import Timer


def get_tfidf_vectorizer(ngram_range: Tuple[int, int], docs_to_train_embedder: List[List[str]]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    vectorizer.fit([' '.join(doc) for doc in docs_to_train_embedder])
    return vectorizer


class BaseTfidfEmbedder(BaseEmbedder):
    def __init__(self, ngram: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        embedder_path = utils.get_model_path(f'{self.embedder_name}.pkl')
        try:
            self.vectorizer: TfidfVectorizer = utils.read_model_from_file(embedder_path)
        except FileNotFoundError:
            self.log.debug('Model not stored. Building TFIDF model from scratch')
            with Timer(f'Building tf-idf for {ngram}-grams', log=self.log):
                self.vectorizer: TfidfVectorizer = get_tfidf_vectorizer((ngram, ngram), self.docs_to_train_embedder)
            utils.write_model_to_file(self.embedder_name, self.vectorizer)

    def embed(self, doc: List[str]):
        return self.vectorizer.transform([' '.join(doc)]).toarray()[0]


class TfidfEmbedderUniGram(BaseTfidfEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class TfidfEmbedderBiGram(BaseTfidfEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class TfidfEmbedderTriGram(BaseTfidfEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)
