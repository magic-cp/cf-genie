from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import variation
from sklearn.decomposition import PCA

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.embedders import EMBEDDERS
from cf_genie.utils import Timer

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():
    log.info('Coefficient of variations (COV) higher than ones means that we should not use MinMaxScaler')
    df = utils.read_cleaned_dataset()
    y = df['most_occurrent_tag_group'].to_numpy()
    print(df.count()['most_ocurrent'])
    unique_y = np.unique(y)
    for embedder in EMBEDDERS:
        log.info('Checking stats for %s', embedder.__name__)
        words = embedder.read_embedded_words()

        log.info('Coefficient of variation is: %s', np.argmax(variation(words, axis=0)))

        # # Reference https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
        # log.info('Testing normality')
        # normalities = [('shapiro', shapiro), ('normaltest', normaltest)]
        # for stat_name, test in normalities:
        #     stat, p = test(words)
        #     log.info('Checking %s', stat_name)
        #     # log.info('Statistics=%s, p=%s', stat, p)
        #     # interpret
        #     alpha = 0.05
        #     if np.all(p > alpha):
        #         log.info('Sample looks Gaussian (fail to reject H0)')
        #     else:
        #         log.info('Sample does not look Gaussian (reject H0)')

        # zs = zscore(words)
        # log.info('Z-score results: %s', zs)
        # for threshold in [1, 2, 3, 4, 5]:
        #     nn = np.where(np.logical_and(zs > threshold, zs <= threshold + 1))
        #     log.info('Z-score values greather than %s: %s', threshold, nn)
        #     log.info('Z-score number of values greater than %s: %s', threshold, nn[0].shape)
        # # log.info('Anderson results: %s', anderson(words))
        # log.info('#' * 80)

        with Timer(f'PCA for {embedder.__name__}', log=log):
            pca = PCA(n_components=2)
            points = pca.fit_transform(words)

        y = np.vectorize(lambda x: 'GEOMETRY' if x == 'GEOMETRY' else 'ADHOC')(y)
        unique_y = np.unique(y)

        plt.figure(figsize=(10, 6))
        for label in unique_y:
            y_label = y == label
            X_label = points[y_label]
            plt.scatter(X_label[:, 0], X_label[:, 1], label=label)
        plt.title(f'PCA components for {embedder.__name__}')
        plt.legend()
        plt.grid()
        utils.write_plot(f'pca/pca-plot-{embedder.__name__}-demo.png', plt)
        plt.close()

        # plt.figure(figsize=(10, 6))
        # plt.hist2d(points[:, 0], points[:, 1], bins=75, cmap='plasma')
        # plt.title(f'Histogram for {embedder.__name__}')
        # plt.colorbar()
        # utils.write_plot(f'histograms/hist2d-{embedder.__name__}.png', plt)
        # plt.close()


if __name__ == '__main__':
    main()
