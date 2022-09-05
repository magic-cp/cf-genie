from itertools import product

import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from cf_genie.embedders.base import BaseEmbedder

import cf_genie.logger as logger
from cf_genie import utils
from cf_genie.embedders import EMBEDDERS
from cf_genie.models import SUPERVISED_MODELS
from cf_genie.models.model_runner import get_model_suffix_name_without_tag

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():
    RELEVANT_SCORES = ['f1_micro', 'hamming_score']
    y_true = utils.read_cleaned_dataset()['most_occurrent_tag_group'].to_numpy()
    y_not_tag_group = y_true != 'ADHOC'

    def get_x(e: BaseEmbedder):
        X = e.read_embedded_words()
        return X[y_not_tag_group]

    y_true = y_true[y_not_tag_group]

    for model_class, embedder_class in product(SUPERVISED_MODELS, EMBEDDERS):
        model = model_class(
            get_x,
            y_true,
            label=get_model_suffix_name_without_tag(embedder_class, 'ADHOC'))
        y_pred = model.predict(get_x(embedder_class([])))

        _, axes = plt.subplots(figsize=(11, 11))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes, normalize='true')
        axes.set_title(f'Confusion matrix for \n {model.model_name}')
        utils.write_plot(f'confusion_matrix/{model.model_name}', plt)
        plt.close()
        log.info('Classification Report for model %s: \n%s', model.model_name, classification_report(y_true, y_pred))
    pass


if __name__ == '__main__':
    main()
