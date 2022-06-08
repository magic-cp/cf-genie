"""
Script to cleanup the dataset. Performs preprocessing over CF statements, and also remapping of tags to "tag groups"
"""
from itertools import groupby
from pprint import PrettyPrinter

import cf_genie.logger as logger
import cf_genie.utils as utils

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


pprint = PrettyPrinter().pprint


log = logger.get_logger(__name__)

GRAPHS = 'GRAPHS'
MATH = 'MATH'
GEOMETRY = 'GEOMETRY'
OPTIMIZATION = 'OPTIMIZATION'
ADHOC = 'ADHOC'
SEARCH = 'SEARCH'

TAG_GROUP_MAPPER = {
    'implementation': ADHOC,
    'data structures': ADHOC,
    'constructive algorithms': ADHOC,
    'brute force': ADHOC,
    'sortings': ADHOC,
    'strings': ADHOC,
    'bitmasks': ADHOC,
    'two pointers': ADHOC,
    'hashing': ADHOC,
    'interactive': ADHOC,
    'string suffix structures': ADHOC,
    '*special': ADHOC,
    'expression parsing': ADHOC,
    'schedules': ADHOC,
    'binary search': SEARCH,
    'ternary search': SEARCH,
    'meet-in-the-middle': SEARCH,
    'geometry': GEOMETRY,
    'graphs': GRAPHS,
    'dfs and similar': GRAPHS,
    'trees': GRAPHS,
    'dsu': GRAPHS,
    'shortest paths': GRAPHS,
    'graph matchings': GRAPHS,
    'math': MATH,
    'number theory': MATH,
    'combinatorics': MATH,
    'probabilities': MATH,
    'matrices': MATH,
    'fft': MATH,
    'chinese remainder theorem': MATH,
    'greedy': OPTIMIZATION,
    'dp': OPTIMIZATION,
    'divide and conquer': OPTIMIZATION,
    'games': OPTIMIZATION,
    'flows': OPTIMIZATION,
    '2-sat': OPTIMIZATION,
}

TAG_GROUPS = set(TAG_GROUP_MAPPER.values())


def main():
    df = utils.read_raw_dataset()

    log.info('Raw dataset shape: %s', df.shape)

    # Let's take a look at the first few rows
    log.info('Sneak peek of the raw dataset:')
    log.info(df.head())

    # few problems don't have any tag neither statement, we remove them
    df = df[df['tags'].notna()]
    df = df[df['statement'].notna()]

    # Tags are split by `;`. We can split them with pandas magic
    df['tags'] = df['tags'].str.split(';', expand=False)

    # Some tags are not really tags, e.g. *<number>. Those tags are not on our
    # mapper, hence we can remove them using it.
    def remove_invalid_tag(tags): return [
        tag for tag in tags if tag in TAG_GROUP_MAPPER]

    def map_tags(tags): return [TAG_GROUP_MAPPER[tag] for tag in tags]
    df['tag_groups'] = df['tags'].apply(
        lambda row: map_tags(remove_invalid_tag(row)))

    # Some problems doesn't map to any tag group, we remove them
    df = df[df['tag_groups'].apply(lambda r: len(r) != 0)]

    # Some statements are empty, so we drop them
    df = df[df['statement'].notna()]

    # Take the tag group that occurrs the most, and expand the dataset
    def max_tag(m):
        m = {k: len(list(v)) for k, v in groupby(m)}
        return max(m, key=lambda k: m[k])
    df['most_occurrent_tag_group'] = df['tag_groups'].apply(
        lambda r: max_tag(r))

    df['preprocessed_statement'] = df['statement'].apply(
        lambda row: utils.preprocess_cf_statement(row))

    # Removing `number` manually from all statements, as it's does'nt really
    # differentiate. We may need to add it later to choose a better primitive. Stuff for later
    df['preprocessed_statement'] = df['preprocessed_statement'].apply(
        lambda row: list(filter(lambda x: x != 'number', row)))

    log.info('Dataset after cleaning and preprocessing:')
    log.info(df.head())
    log.info('Dataset shape after cleaning: %s', df.shape)


    for tag_group in TAG_GROUPS:
        df_tag_group = df[df['most_occurrent_tag_group'] == tag_group]
        print(f'Working on {tag_group}')
        utils.plot_wordcloud(
            ' '.join(
                list(
                    df_tag_group['preprocessed_statement'].str.join(' '))),
            plot_title=tag_group,
            file_name=tag_group +
            '.png')

    utils.write_dataframe_to_csv('dataset_cleaned.csv', df)


if __name__ == '__main__':
    main()
