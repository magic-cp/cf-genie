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

    def tagged_doc(r): return TaggedDocument(
        words=r['preprocessed_statement'].split(' '), tags=[
            r.most_occurrent_tag_group])
    train_tagged = train.apply(tagged_doc, axis=1)
    test_tagged = test.apply(tagged_doc, axis=1)

    log.info('Sneak peek of train_tagged')
    log.info(train_tagged.head())

    # Initializtion of the model
    model = Doc2Vec(vector_size=50, negative=5, hs=0, min_count=2, sample=0, workers=cores, epochs=40)
    model.build_vocab([x for x in tqdm(train_tagged.values)])

    log.info(f"Word 'point' appeared {model.wv.get_vecattr('point', 'count')} times in the training corpus.")

    with Timer('Word2Vec training', log=log):
        model.train(train_tagged, total_examples=model.corpus_count, epochs=model.epochs)

    log.info('Demo infering: %s', model.infer_vector(['point', 'number']))

    def get_ranks(tagged_docs):
        ranks = []
        second_ranks = []
        for doc_id in tagged_docs.index:
            log.debug('doc_id: %s', doc_id)
            inferred_vector = model.infer_vector(tagged_docs[doc_id].words)
            log.debug('Inferred vector for doc id #%s: %s', doc_id, inferred_vector)
            sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
            log.debug('Similarity scores for doc id #%s: %s', doc_id, sims)
            rank = [docid for docid, sim in sims].index(tagged_docs[doc_id].tags[0])
            ranks.append((doc_id, rank))

            second_ranks.append(sims[1][0])
        return ranks, second_ranks

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

    ranks, _ = get_ranks(train_tagged)
    log.info('Plotting for principal ranks on training data')
    plot_rank_distribution([r[1] for r in ranks],
                           'pie_chart_most_similar_tags_on_training_data.png',
                           'Ranking of most similar tags on training data')

    test_ranks, _ = get_ranks(test_tagged)
    log.info('Plotting for principal ranks on test data')
    plot_rank_distribution([r[1] for r in test_ranks],
                           'pie_chart_most_similar_tags_on_test_data.png',
                           'Ranking of most similar tags on test data')

    # Let's pick a random document from the train and test data and see what the model predicts.
    train_idx = random.choice(train_tagged.index)
    test_idx = random.choice(test_tagged.index)

    log.info('Train idx: %s', train_idx)
    log.info('Train Problem id: %s%s', df.iloc[train_idx].contest_id, df.iloc[train_idx].problem_id)

    log.info('Tag group for train idx: %s', df.iloc[train_idx].most_occurrent_tag_group)

    inferred_vector = model.infer_vector(train_tagged[train_idx].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    log.info('Predictions of tag group for train idx: %s', sims)

    log.info('Test idx: %s', test_idx)
    log.info('Test Problem id: %s%s', df.iloc[test_idx].contest_id, df.iloc[test_idx].problem_id)

    log.info('Tag group for test idx: %s', df.iloc[test_idx].most_occurrent_tag_group)

    inferred_vector = model.infer_vector(test_tagged[test_idx].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    log.info('Predictions of tag group for test idx: %s', sims)


if __name__ == '__main__':
    main()
