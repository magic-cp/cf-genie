from itertools import product

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import cf_genie.logger as logger
from cf_genie import utils
from cf_genie.embedders import EMBEDDERS
from cf_genie.models import SUPERVISED_MODELS, TrainingMethod
from cf_genie.models.model_analyisis import \
    get_acc_pandas_df_for_model_all_classes
from cf_genie.models.model_runner import get_model_suffix_name_for_all_classes
import matplotlib.pyplot as plt

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():
    RELEVANT_SCORES = ['f1_micro', 'hamming_score']
    y_true = utils.read_cleaned_dataset()['most_occurrent_tag_group'].to_numpy()
    for model_class, embedder_class in product(SUPERVISED_MODELS, EMBEDDERS):
        model = model_class(
            embedder_class.read_embedded_words,
            y_true,
            TrainingMethod.GRID_SEARCH_CV,
            label=get_model_suffix_name_for_all_classes(embedder_class))
        y_pred = model.predict(embedder_class.read_embedded_words())

        _, axes = plt.subplots(figsize=(11, 11))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes)
        axes.set_title(f'Confusion matrix for \n {model.model_name}')
        utils.write_plot(f'confusion_matrix/{model.model_name}', plt)
        plt.close()
        log.info('Classification Report for model %s: \n%s', model.model_name, classification_report(y_true, y_pred))
    pass


if __name__ == '__main__':
    main()
