"""
Train the doc2vec model.

Reference; https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
"""
import collections
import random
from time import time

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils as sklearn_utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import cf_genie.logger as logger
import cf_genie.utils as utils
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

    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['most_occurrent_tag_group'])

    def tagged_doc(r):
        return TaggedDocument(words=r['preprocessed_statement'].split(' '), tags=[r.name])
    train_tagged = train.apply(tagged_doc, axis=1)
    test_tagged = test.apply(tagged_doc, axis=1)

    log.info('Sneak peek of train_tagged')
    log.info(train_tagged.head())

    # Initializtion of the model
    model = Doc2Vec(vector_size=50, negative=5, hs=0, min_count=2, sample=0, workers=cores, epochs=40)
    model.build_vocab(train_tagged.values)

    log.info(f"Word 'point' appeared {model.wv.get_vecattr('point', 'count')} times in the training corpus.")

    with Timer('Word2Vec training', log=log):
        model.train(train_tagged, total_examples=model.corpus_count, epochs=model.epochs)

    log.info('Demo infering: %s', model.infer_vector(['point', 'number']))

    def get_ranks(tagged_docs, model):
        ranks = []
        with Timer("Doc2Vec inferring", log=log):
            for doc_id in tqdm(tagged_docs.index, desc='Doc2Vec inferring'):
                log.debug('doc_id: %s', doc_id)
                inferred_vector = model.infer_vector(tagged_docs[doc_id].words)
                log.debug('Inferred vector for doc id #%s: %s', doc_id, inferred_vector)
                sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
                log.debug('Similarity scores for doc id #%s: %s', doc_id, sims)
                rank = [docid for docid, sim in sims].index(doc_id)
                ranks.append(rank)
        return ranks

    def plot_rank_distribution(ranks, file_name, plot_title):
        # Values near 0 indicate that the inferred vector is very similar to the tag.
        counter = collections.Counter(ranks)
        log.info('Ranks: %s', counter)
        labels = sorted(counter)
        sizes = [v for v in counter.values()]
        log.info('Labels: %s', labels)
        log.info('Sizes: %s', sizes)
        utils.plot_pie_chart(labels, sizes, file_name, plot_title)

        total_sum = sum(sizes)
        log.info('The model predicted the primary rank with an accuracy of %.2f', 100 * counter.get(0) / total_sum)

    ranks = get_ranks(train_tagged, model)
    log.info('Plotting for principal ranks on training data')
    plot_rank_distribution(ranks,
                           'pie_chart_most_similar_tags_on_training_data.png',
                           'Ranking of most similar tags on training data')

    tagged_docs = df.apply(tagged_doc, axis=1)
    modelall = Doc2Vec(vector_size=50, negative=5, hs=0, min_count=2, sample=0, workers=cores, epochs=40)

    modelall.build_vocab(tagged_docs.values)

    log.info(f"Word 'point' appeared {modelall.wv.get_vecattr('point', 'count')} times in the entire corpus.")

    with Timer('Word2Vec training with all data', log=log):
        modelall.train(tagged_docs, total_examples=modelall.corpus_count, epochs=modelall.epochs)

    log.info('Demo infering: %s', modelall.infer_vector(['point', 'number']))

    ranks = get_ranks(tagged_docs, modelall)
    log.info('Plotting for principal ranks on all data')
    plot_rank_distribution(ranks,
                           'pie_chart_most_similar_tags_on_all_data.png',
                           'Ranking of most similar tags on all data')


if __name__ == '__main__':
    main()
