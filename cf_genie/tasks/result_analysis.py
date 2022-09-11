from itertools import product

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import cf_genie.logger as logger
from cf_genie import utils
from cf_genie.embedders import EMBEDDERS
from cf_genie.embedders.base import BaseEmbedder
from cf_genie.models import SUPERVISED_MODELS
from cf_genie.models.model_runner import get_model_suffix_name_without_tag

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():
    y_true = utils.read_cleaned_dataset('without-adhoc-test')['most_occurrent_tag_group'].to_numpy()


    for model_class, embedder_class in product(SUPERVISED_MODELS, EMBEDDERS):
        embedder = embedder_class([], 'without-adhoc-test')
        model = model_class(
            embedder.read_embedded_words,
            y_true,
            label=get_model_suffix_name_without_tag(embedder_class, 'ADHOC'))
        y_pred = model.predict(embedder.read_embedded_words())

        for normalize in ['true', 'pred', 'all', None]:
            _, axes = plt.subplots(figsize=(11, 11))
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes, normalize=normalize)
            axes.set_title(f'Confusion matrix for \n {model.model_name} \n with {normalize} normalization')
            utils.write_plot(f'confusion_matrix/{model.model_name}/normalize-{normalize}', plt)
            plt.close()
        log.info('Classification Report for model %s: \n%s', model.model_name, classification_report(y_true, y_pred))
    pass


if __name__ == '__main__':
    main()
