import pickle

import pandas as pd
from gensim.models import FastText
from hyperopt import STATUS_OK

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.utils.timer import Timer

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def model_fn(df: pd.DataFrame):
    def wrapped():
        model = FastText(epochs=30, workers=utils.CORES)
        model.build_vocab(df['preprocessed_statement'].values)
        log.info('FastText model: %s', model)

        with Timer(f'FastText training {model}', log=log):
            model.train(
                df['preprocessed_statement'].values,
                total_examples=model.corpus_count,
                epochs=model.epochs,
                compute_loss=True)

        # return {
        #     'loss': -1,
        #     'status': STATUS_OK,
        #     'attachments': {
        #         'model': pickle.dumps(model)
        #     }
        # }
        return model
    return wrapped


def main():
    log.info('Generating word2vec model...')
    df = utils.read_cleaned_dataset()

    log.info('Dataset:')
    log.info(df.head())

    cores = utils.get_num_of_cores()
    df['preprocessed_statement'] = df['preprocessed_statement'].apply(lambda x: x.split(' '))
    log.info('fastText model will use %s cores', cores)

    model = model_fn(df)()
    vec = model.wv.get_sentence_vector(df.iloc[0].preprocessed_statement)
    log.info('Vector for first statement: %s', vec)
    sims = model.wv.similar_by_vector(vec, topn=10)
    log.info('Similar statements: %s', sims)

    vec = model.wv.get_sentence_vector(df.iloc[1].preprocessed_statement)
    log.info('Vector for second statement: %s', vec)
    log.info('')
    sims = model.wv.similar_by_vector(vec, topn=10)
    log.info('Similar statements: %s', sims)
    pass


if __name__ == '__main__':
    main()
