from itertools import product

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import cf_genie.logger as logger
from cf_genie import utils
from cf_genie.embedders import EMBEDDERS
from cf_genie.embedders.base import BaseEmbedder
from cf_genie.models import SUPERVISED_MODELS
from cf_genie.models.model_runner import get_model_suffix_name_for_all_classes

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():

    REPORT_CONFIG = [
        # Uncomment to test the imbalanced case
        # {
        #     'test_dataset': 'without-adhoc-test',
        #     'training_dataset': 'without-adhoc-train',
        # },
        {
            'test_dataset': 'test',
            'training_dataset': 'train'
        }
    ]
    for report_config, model_class, embedder_class in product(REPORT_CONFIG, SUPERVISED_MODELS, EMBEDDERS):
        y_true = utils.read_cleaned_dataset(report_config['test_dataset'])['most_occurrent_tag_group'].to_numpy()
        embedder = embedder_class([], label=report_config['test_dataset'], training_label=report_config['training_dataset'])
        model = model_class(
            embedder.read_embedded_words,
            y_true,
            label=get_model_suffix_name_for_all_classes(embedder.embedder_name))
        y_pred = model.predict(embedder.read_embedded_words())

        for normalize in ['true', None]:
            _, axes = plt.subplots(figsize=(8, 8))
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes, normalize=normalize)
            axes.set_title(f'Confusion matrix for \n {model.model_name} \n with {normalize} normalization')
            plt.tight_layout()
            utils.write_plot(f'confusion_matrix/{model.model_name}/normalize-{normalize}', plt)
            plt.close()
        log.info('Classification Report for model %s: \n%s', model.model_name, classification_report(y_true, y_pred))
    pass


if __name__ == '__main__':
    main()
