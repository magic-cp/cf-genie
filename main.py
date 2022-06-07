import csv

import logger

logger.setup_applevel_logger(is_debug=False, file_name='app.logs', simple_logs=False)

from pprint import PrettyPrinter

pprint = PrettyPrinter().pprint


import utils

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

    # Drop unnecessary columns kept on the original CSV for documentation purposes
    df.drop(columns=['is_interactive', 'input_spec', 'output_spec', 'url'], axis=1, inplace=True)

    # Let's take a look at the first few rows
    print(df.head())

    # few problems don't have any tag neither statement, we remove them
    df = df[df['tags'].notna()]
    df = df[df['statement'].notna()]

    # Tags are split by `;`. We can split them with pandas magic
    df['tags'] = df['tags'].str.split(';', expand=False)

    # Some tags are not really tags, e.g. *<number>. Those tags are not on our mapper, hence we can remove them using it.
    remove_invalid_tag = lambda tags: [tag for tag in tags if tag in TAG_GROUP_MAPPER]
    map_tags = lambda tags: sorted([TAG_GROUP_MAPPER[tag] for tag in tags])
    df['tag_group'] = df['tags'].apply(lambda row: map_tags(remove_invalid_tag(row)))

    # Some statements are empty, so we drop them
    df = df[df['statement'].isna() == False]

    print(df.head())

    log.info('Dataset shape after cleaning: %s', df.shape)

    df['statement'] = df['statement'].apply(lambda row: ' '.join(utils.preprocess_cf_statement(row)))

    print(df.head())

    for tag_group in TAG_GROUPS:
        print(tag_group)
    utils.plot_wordcloud(' '.join(list(df['statement'].values)))

if __name__ == '__main__':
    main()
