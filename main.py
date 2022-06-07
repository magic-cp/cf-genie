import csv

import logger

logger.setup_applevel_logger(is_debug=False, file_name='app.logs', simple_logs=False)

from pprint import PrettyPrinter

pprint = PrettyPrinter().pprint


import utils

log = logger.get_logger(__name__)

def main():
    df = utils.read_raw_dataset()

    log.info('Raw dataset shape: %s', df.shape)

    # Drop unnecessary columns kept on the original CSV for documentation purposes
    df.drop(columns=['is_interactive', 'input_spec', 'output_spec'], axis=1, inplace=True)

    # Some statements are empty, so we drop them
    df = df[df['statement'].isna() == False]

    print(df.head())

    log.info('Dataset shape after cleaning: %s', df.shape)

    df['statement'] = df['statement'].apply(lambda row: ' '.join(utils.preprocess_cf_statement(row)))

    print(df.head())

    utils.plot_wordcloud(' '.join(list(df['statement'].values)))

if __name__ == '__main__':
    main()
