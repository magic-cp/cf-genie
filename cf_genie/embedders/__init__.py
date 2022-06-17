"""
Exports all embedders relevant to our proyect
"""
from typing import Callable, List, Type

from cf_genie.embedders.base import BaseEmbedder
from cf_genie.embedders.doc2vec import Doc2VecEmbedder
from cf_genie.embedders.fasttext import FastTextEmbedder
from cf_genie.embedders.tfidf import (TfidfEmbedderBiGram,
                                      TfidfEmbedderTriGram,
                                      TfidfEmbedderUniGram)

EMBEDDERS: List[Type[BaseEmbedder]] = [Doc2VecEmbedder, FastTextEmbedder,
                                       TfidfEmbedderUniGram, TfidfEmbedderBiGram, TfidfEmbedderTriGram]
