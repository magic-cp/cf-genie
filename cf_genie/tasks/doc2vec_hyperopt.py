"""
Run hyperopt on doc2vec to see what give us our best parameters
"""

import collections

import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.mongoexp import MongoTrials
from tqdm import tqdm

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.utils import Timer

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


SPACE = {
    'vector_size': hp.choice('vector_size', [50, 100, 200, 300]),
    'dm': hp.choice('dm', [0, 1]),
    'window': hp.choice('window', [4, 5]),
    'hs': hp.choice('hs', [0, 1]),
}


def objective(tagged_docs: pd.DataFrame):
    def fn(params):
        log.info('Building word2vec model with params: %s', params)
        model = Doc2Vec(**params, negative=5, min_count=2, sample=0, workers=utils.CORES, epochs=40)

        model.build_vocab(tagged_docs.values)

        with Timer(f'Word2Vec training {model}', log=log):
            model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

        ranks = []
        with Timer(f"Doc2Vec inferring {model}", log=log):
            for doc_id in tqdm(tagged_docs.index, desc='Doc2Vec inferring'):
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
            'params': params,
        }

    return fn


def main():
    trials = Trials()
    log.info('Generating word2vec model...')
    df = utils.read_cleaned_dataset()

    log.info('Dataset:')
    log.info(df.head())

    cores = utils.get_num_of_cores()
    log.info('Doc2Vec model will use %s cores', cores)

    def tagged_doc(r): return TaggedDocument(words=r['preprocessed_statement'].split(' '), tags=[r.name])
    tagged_docs = df.apply(tagged_doc, axis=1)

    trials = MongoTrials('mongo://localhost:27017/admin/jobs', exp_key='exp1')
    with Timer('Hyperopt search for best parameters for Doc2Vec', log=log):
        best_params = fmin(
            objective(tagged_docs),
            SPACE,
            algo=tpe.suggest,
            trials=trials,
            show_progressbar=True,
            max_evals=40)

    log.info('Best parameters found %s', space_eval(SPACE, best_params))
    log.info('Type of best_params: %s', type(best_params))
    log.info('Trials: %s', trials)
    log.info('Space: %s', SPACE)


if __name__ == '__main__':
    main()
