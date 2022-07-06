"""
Script to cleanup the dataset. Performs preprocessing over CF statements, and also remapping of tags to "tag groups"
"""
from itertools import groupby

from tqdm import tqdm

import cf_genie.logger as logger
import cf_genie.utils as utils

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)

tqdm.pandas()


def main():
    df = utils.read_raw_dataset()

    log.info('Raw dataset shape: %s', df.shape)

    # Let's take a look at the first few rows
    log.info('Sneak peek of the raw dataset:')
    log.info(df.head())

    # few problems don't have any tag neither statement, we remove them
    df = df[df['tags'].notna()]
    df = df[df['statement'].notna()]

    def remove_invalid_tag(tags): return [
        tag for tag in tags if tag in utils.TAG_GROUP_MAPPER]

    # Tags are split by `;`. We can split them with pandas magic
    df['cleaned_tags'] = df['tags'].progress_apply(lambda tags: ';'.join(remove_invalid_tag(tags.split(';'))))
    df = df[df['cleaned_tags'] != '']
    log.info('Head of tags: %s', df['tags'].head())

    # Some tags are not really tags, e.g. *<number>. Those tags are not on our
    # mapper, hence we can remove them using it.

    def map_tags(tags):
        return [utils.TAG_GROUP_MAPPER[tag] for tag in tags]
    df['tag_groups'] = df['cleaned_tags'].progress_apply(lambda row: map_tags(row.split(';')))

    # Some problems doesn't map to any tag group, we remove them
    df = df[df['tag_groups'].progress_apply(lambda r: len(r) != 0)]

    # Some statements are empty, so we drop them
    df = df[df['statement'].notna()]

    # Take the tag group that occurrs the most, and expand the dataset
    def max_tag(m):
        m = {k: len(list(v)) for k, v in groupby(m)}
        return max(m, key=lambda k: m[k])
    df['most_occurrent_tag_group'] = df['tag_groups'].progress_apply(
        lambda r: max_tag(r))

    # Preprocess each statement
    df['preprocessed_statement'] = df['statement'].progress_apply(
        lambda row: utils.preprocess_cf_statement(row))

    # Removing `number` manually from all statements, as it's does'nt really
    # differentiate. We may need to add it later to choose a better primitive. Stuff for later
    df['preprocessed_statement'] = df['preprocessed_statement'].progress_apply(
        lambda row: list(filter(lambda x: x != 'number', row)))

    # Converting as a string separated by ' '
    df['preprocessed_statement'] = df['preprocessed_statement'].progress_apply(lambda row: ' '.join(row))

    # Same for the tag groups
    df['tag_groups'] = df['tag_groups'].progress_apply(lambda row: ' '.join(row))

    log.info('Dataset after cleaning and preprocessing:')
    log.info(df.head())
    log.info('Dataset shape after cleaning: %s', df.shape)

    log.info('Dtypes: %s', df.dtypes['preprocessed_statement'])
    utils.write_cleaned_dataframe_to_csv(df)


if __name__ == '__main__':
    main()
