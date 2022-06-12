import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import cf_genie.logger as logger
import cf_genie.utils as utils

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():
    df = utils.read_cleaned_dataset()

    df = df[['cleaned_tags']]

    log.info('Sneak peak: %s', df.head())

    df['cleaned_tags'] = df['cleaned_tags'].apply(lambda row: set(row.split(';')))

    all_tags = set()
    for tags in df['cleaned_tags']:
        all_tags.update(tags)

    log.info('All tags: %s', all_tags)

    for tag in all_tags:
        df[tag] = df['cleaned_tags'].apply(lambda row: tag in row)

    df.drop(['cleaned_tags'], axis=1, inplace=True)

    corr = df.corr()
    plt.figure(figsize=(10, 10))

    sns.heatmap(corr, cmap='BrBG').set_title('Correlation matrix for tags')
    # plt.title('Correlation between tags')
    plt.tight_layout()
    utils.write_plot('correlations/correlation_matrix_tags.png', plt)

    top = 5
    for tag in corr:
        correlations_df = corr[[tag]]
        log.info('Correlations df: %s', correlations_df)
        correlations = sorted(corr[tag].items(), key=lambda x: x[1])
        log.info('Tag in corr: %s', tag)
        log.info('Top %s positive correlations: %s', top, sorted(correlations[-top:], key=lambda x: x[1], reverse=True))
        log.info('Top %s negative correlations: %s', top, correlations[:top])
        log.info('')
        plt.figure(figsize=(9, 5))
        sns.heatmap(correlations_df.sort_values(by=tag, ascending=False).T, cmap='BrBG', annot=True,
                    annot_kws={'rotation': 90}).set_title(f'Correlation matrix for tag "{tag}"')
        plt.tight_layout()
        utils.write_plot(f'correlations/correlation_matrix_for__{tag.replace(" ", "_")}.png', plt)
        # plt.show()

    # plt.show()


if __name__ == '__main__':
    main()
