from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

import cf_genie.logger as logger
from cf_genie.embedders import *
from cf_genie.models import KMeansClustering
from cf_genie.utils import Timer
from cf_genie.utils.read_write_files import read_cleaned_dataset

logger.setup_applevel_logger(is_debug=False, file_name=__file__, simple_logs=True)

log = logger.get_logger(__name__)

tqdm.pandas()


def main():
    df = read_cleaned_dataset()
    words = TfidfEmbedderUniGram.read_embedded_words()
    kmeans = KMeansClustering(words, k=2)
    log.info('KMeans results: %s', kmeans)
    log.info('Labels of training data: %s', kmeans._model.labels_)
    labels = kmeans.predict(words)
    unique_labels = np.unique(labels)
    cleaned_tags = df['cleaned_tags'].str.split(';').apply(set)
    log.info('Cleaned tags: %s', cleaned_tags)

    with Timer(f'PCA tests with KMeans', log=log):
        pca = PCA(n_components=2)
        points = pca.fit_transform(words)

    plt.figure(figsize=(10, 6))
    for label in unique_labels:
        label_criteria = labels == label
        X_label = points[label_criteria]
        plt.scatter(X_label[:, 0], X_label[:, 1], label=label)
    plt.title(f'PCA components for KMeans tests')
    plt.legend()
    plt.grid()
    plt.show()
    # utils.write_plot(f'pca/pca-plot-{embedder.__name__}-demo.png', plt)
    plt.close()

    for label in unique_labels:
        label_criteria = labels == label
        tags_set = set()
        tags_counter = Counter()
        for tags in cleaned_tags.loc[label_criteria]:
            tags_set = tags_set.union(tags)
            tags_counter.update(tags)
        tag_groups_set = set()
        tag_group_counter = Counter()
        for tag_group in df['most_occurrent_tag_group'].loc[label_criteria]:
            tag_groups_set.add(tag_group)
            tag_group_counter.update([tag_group])
        # X_label = points[label_criteria]
        log.info('Tags present in cluster %s : %s', label, tags_set)
        log.info('Tags counter in cluster %s : %s', label, tags_counter)
        log.info('Tag groups present in cluster %s : %s', label, tag_groups_set)
        log.info('Tag groups counter in cluster %s : %s', label, tag_group_counter)

    pass


if __name__ == '__main__':
    main()
