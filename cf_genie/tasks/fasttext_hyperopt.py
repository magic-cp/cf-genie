import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.embedders import FastTextEmbedder
from cf_genie.utils.timer import Timer

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():
    log.info('Generating word2vec model...')
    df = utils.read_cleaned_dataset()

    with Timer('Loading FastTextEmbedder'):
        embedder = FastTextEmbedder(df['preprocessed_statement'].values)

    log.info('Testing around first statement: %s', embedder.embed(df.iloc[0].preprocessed_statement))
    pass


if __name__ == '__main__':
    main()
