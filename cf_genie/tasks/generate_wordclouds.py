import pandas as pd

import cf_genie.logger as logger
import cf_genie.utils as utils

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():
    log.info('Loading dataset...')
    df = utils.read_cleaned_dataset()
    log.info(df.head())

    for tag_group in utils.TAG_GROUPS:
        df_tag_group = df[df['most_occurrent_tag_group'] == tag_group]
        log.info(f'Working on {tag_group}')
        text = ' '.join(
            df_tag_group['preprocessed_statement'].str.join(' ').values)
        log.debug(text)
        utils.plot_wordcloud(
            text,
            plot_title='Nube de palabras para ' + tag_group,
            file_name='wordclouds/' + 'wordcloud-' + tag_group +
            '.png')


if __name__ == '__main__':
    main()
