"""
Script to cleanup the dataset. Performs preprocessing over CF statements, and also remapping of tags to "tag groups"
"""

from imblearn.over_sampling import RandomOverSampler

import cf_genie.logger as logger
import cf_genie.utils as utils

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():
    name_suffix = 'without-adhoc-train'
    df = utils.read_cleaned_dataset(name_suffix=name_suffix)

    log.info('Initial dataset shape: %s', df.shape)

    log.info('Initial dataset: %s', df.head())

    log.info('Value counts of initial dataset:\n%s', df['most_occurrent_tag_group'].value_counts())
    log.info('Percentages of tags of initial dataset:\n%s', df['most_occurrent_tag_group'].value_counts(normalize=True))

    ros = RandomOverSampler(random_state=42)
    df_resampled, _ = ros.fit_resample(df, df['most_occurrent_tag_group'])

    log.info('Over-sampled dataset shape: %s', df.shape)
    log.info('Value counts of dataset after over-sampling:\n%s',
             df_resampled['most_occurrent_tag_group'].value_counts())
    log.info('Percentages of tags of initial dataset over-sampling:\n%s',
             df_resampled['most_occurrent_tag_group'].value_counts(normalize=True))

    utils.write_cleaned_dataframe_to_csv(df_resampled, name_suffix=name_suffix + '-balanced')


if __name__ == '__main__':
    main()
