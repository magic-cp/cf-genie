"""
Run hyperopt on doc2vec to see what give us our best parameters
"""

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.embedders import Doc2VecEmbedder
from cf_genie.utils import Timer

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

    with Timer('Hyperopt search for best parameters for Doc2Vec', log=log):
        embedder = Doc2VecEmbedder(df['preprocessed_statement'].to_numpy())
        hyperopt_info = embedder.hyperopt_info

    log.info('Best parameters found %s', hyperopt_info.best_params_evaluated_space)

    log.info('Best trial model: %s', embedder.model)


if __name__ == '__main__':
    main()
