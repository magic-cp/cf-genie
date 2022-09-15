"""
Script to split the training and test split
"""
import argparse

from sklearn.model_selection import train_test_split

import cf_genie.logger as logger
import cf_genie.utils as utils

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)

def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Splits the dataset into training and test split')
    parser.add_argument('--test-size', type=float, help='Test percentage size as a float', default=0.40)
    return parser.parse_args(args)

def main(*args):
    args = parse_args(args)
    name_suffix = 'without-adhoc'
    df = utils.read_cleaned_dataset(name_suffix=name_suffix)

    log.info('Raw dataset shape: %s', df.shape)

    X = df.index
    y = df['most_occurrent_tag_group']
    df_train, df_test = train_test_split(df, test_size=args.test_size, random_state=42, stratify=y)

    log.info('Train dataset shape: %s', df_train.shape)
    log.info('Train dataset: %s', df_train.head())
    log.info('Value counts of train dataset: %s', df_train['most_occurrent_tag_group'].value_counts())
    log.info('Value counts of train dataset as percentages: %s',
             df_train['most_occurrent_tag_group'].value_counts(normalize=True))
    log.info('')

    log.info('Test dataset shape: %s', df_test.shape)
    log.info('Test dataset: %s', df_test.head())
    log.info('Value counts of test dataset: %s', df_test['most_occurrent_tag_group'].value_counts())
    log.info('Value counts of test dataset as percentages: %s',
             df_test['most_occurrent_tag_group'].value_counts(normalize=True))

    for df_ in [df_train, df_test]:
        df_['preprocessed_statement'] = df_['preprocessed_statement'].apply(lambda row: ' '.join(row))

        # Same for the tag groups
        df_['tag_groups'] = df_['tag_groups'].apply(lambda row: ' '.join(row))

    utils.write_cleaned_dataframe_to_csv(df_test, name_suffix=name_suffix + '-test')
    utils.write_cleaned_dataframe_to_csv(df_train, name_suffix=name_suffix + '-train')


if __name__ == '__main__':
    main()
