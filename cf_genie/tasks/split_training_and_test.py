"""
Script to cleanup the dataset. Performs preprocessing over CF statements, and also remapping of tags to "tag groups"
"""
from itertools import groupby

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import cf_genie.logger as logger
import cf_genie.utils as utils
import argparse

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)

tqdm.pandas()


def main():
    name_suffix = 'without-adhoc'
    df = utils.read_cleaned_dataset(name_suffix=name_suffix)

    log.info('Raw dataset shape: %s', df.shape)

    X = df.index
    y = df['most_occurrent_tag_group']
    df_train, df_test = train_test_split(df, test_size=0.20, random_state=42, stratify=y)

    log.info('Train dataset shape: %s', df_train.shape)
    log.info('Train dataset: %s', df_train.head())
    log.info('Value counts of train dataset: %s', df_train['most_occurrent_tag_group'].value_counts())
    log.info('Value counts of train dataset as percentages: %s', df_train['most_occurrent_tag_group'].value_counts(normalize=True))
    log.info('')

    log.info('Test dataset shape: %s', df_test.shape)
    log.info('Test dataset: %s', df_test.head())
    log.info('Value counts of test dataset: %s', df_test['most_occurrent_tag_group'].value_counts())
    log.info('Value counts of test dataset as percentages: %s', df_test['most_occurrent_tag_group'].value_counts(normalize=True))


    utils.write_cleaned_dataframe_to_csv(df, name_suffix=name_suffix)


if __name__ == '__main__':
    main()
