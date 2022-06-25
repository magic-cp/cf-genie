"""
Exports all embedders relevant to our proyect
"""
from typing import List, Type

from cf_genie.embedders.base import BaseEmbedder
from cf_genie.embedders.doc2vec import (Doc2VecEmbedder, Doc2VecEmbedder30,
                                        Doc2VecEmbedder50, Doc2VecEmbedder100,
                                        Doc2VecEmbedder150, Doc2VecEmbedder200)
from cf_genie.embedders.fasttext import (FastTextEmbedder, FastTextEmbedder30,
                                         FastTextEmbedder50,
                                         FastTextEmbedder100,
                                         FastTextEmbedder150,
                                         FastTextEmbedder200)
from cf_genie.embedders.tfidf import (TfidfEmbedderBiGram,
                                      TfidfEmbedderTriGram,
                                      TfidfEmbedderUniGram)

EMBEDDERS: List[Type[BaseEmbedder]] = [
    # Doc2vec variations
    Doc2VecEmbedder,
    Doc2VecEmbedder30,
    Doc2VecEmbedder50,
    Doc2VecEmbedder100,
    Doc2VecEmbedder150,
    Doc2VecEmbedder200,

    # FastText variations
    FastTextEmbedder,
    FastTextEmbedder30,
    FastTextEmbedder50,
    FastTextEmbedder100,
    FastTextEmbedder150,
    FastTextEmbedder200,

    # tf-idf variations
    TfidfEmbedderUniGram,
]  # TfidfEmbedderBiGram]  # , TfidfEmbedderTriGram]
