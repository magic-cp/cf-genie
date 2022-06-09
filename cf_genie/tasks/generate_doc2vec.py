"""
Train the doc2vec model.

Reference; https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
"""
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils as sklearn_utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import cf_genie.logger as logger
import cf_genie.utils as utils

from time import time

import collections

from cf_genie.utils.timer import Timer

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():
    log.info('Generating word2vec model...')
    df = utils.read_cleaned_dataset()


    log.info('Dataset:')
    log.info(df.head())

    cores = utils.get_num_of_cores()
    log.info('Doc2Vec model will use %s cores', cores)

    train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['most_occurrent_tag_group'])

    tagged_doc = lambda r: TaggedDocument(words=r['preprocessed_statement'].split(' '), tags=[r.most_occurrent_tag_group])
    train_tagged = train.apply(tagged_doc,        axis=1)
    test_tagged = test.apply(tagged_doc,        axis=1)

    log.info('Sneak peek of train_tagged')
    log.info(train_tagged.head())

    # Initializtion of the model
    model = Doc2Vec(vector_size=50, negative=5, hs=0, min_count=2, sample=0, workers=cores, epochs=40)
    model.build_vocab([x for x in tqdm(train_tagged.values)])

    log.info(f"Word 'point' appeared {model.wv.get_vecattr('point', 'count')} times in the training corpus.")

    with Timer('Word2Vec training', log=log):
        model.train(train_tagged, total_examples=model.corpus_count, epochs=model.epochs)

    log.info('Demo infering: %s', model.infer_vector(['point', 'number']))

    ranks = []
    second_ranks = []
    for doc_id in train_tagged.index:
        log.debug('doc_id: %s', doc_id)
        inferred_vector = model.infer_vector(train_tagged[doc_id].words)
        log.debug('Inferred vector for doc id #%s: %s', doc_id, inferred_vector)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        log.debug('Similarity scores for doc id #%s: %s', doc_id, sims)
        rank = [docid for docid, sim in sims].index(train_tagged[doc_id].tags[0])
        ranks.append(rank)

        second_ranks.append(sims[1])

    # Values near 0 indicate that the inferred vector is very similar to the tag.
    counter = collections.Counter(ranks)
    log.info('Ranks: %s', counter)
    utils.plot_pie_chart(sorted(counter), [v for v in counter.values()], 'pie_chart_most_similar_tags.png', 'Ranking of most similar tags')

if __name__ == '__main__':
    main()
