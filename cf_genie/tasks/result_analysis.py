from itertools import product

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

import cf_genie.logger as logger
from cf_genie import utils
from cf_genie.embedders import EMBEDDERS
from cf_genie.embedders.base import BaseEmbedder
from cf_genie.models import SUPERVISED_MODELS
from cf_genie.models.model_runner import get_model_suffix_name_for_all_classes

import pandas as pd
import numpy as np

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():

    REPORT_CONFIG = [
        {
            'test_dataset': 'without-adhoc-test',
            'training_dataset': 'without-adhoc-train-balanced',
        },
        {
            'test_dataset': 'test',
            'training_dataset': 'train'
        }
    ]
    confusion_matrix_paths = {
        'without-adhoc-train-balanced': [],
        'train': []
    }
    for report_config, model_class, embedder_class in product(REPORT_CONFIG, SUPERVISED_MODELS, EMBEDDERS):
        y_true = utils.read_cleaned_dataset(report_config['test_dataset'])['most_occurrent_tag_group'].to_numpy()
        embedder = embedder_class([], label=report_config['test_dataset'], training_label=report_config['training_dataset'])
        model = model_class(
            embedder.read_embedded_words,
            y_true,
            label=get_model_suffix_name_for_all_classes(embedder.embedder_name))
        y_pred = model.predict(embedder.read_embedded_words())

        for normalize in ['true', None]:
            _, axes = plt.subplots(figsize=(6, 6))
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axes, normalize=normalize, colorbar=False)
            normalize_msg = 'normalized' if normalize else ''
            # axes.set_title(f'Matriz de confusión para \n {model.display_name} usando {embedder.display_name} \n {normalize_msg}')
            caption = f'Matriz de confusión para \n {model.display_name} usando {embedder.display_name} en el conjunto de datos \\textit{{{report_config["training_dataset"]}}} \n {normalize_msg}'
            label = f'{model.model_name}-normalize-{normalize}'.lower().replace(' ', '-')
            plt.tight_layout()
            p = f'confusion_matrix/{model.model_name}-normalize-{normalize}.png'
            utils.write_plot(p, plt)

            if not normalize:
                confusion_matrix_paths[report_config['training_dataset']].append((p, caption.replace('\n', ' '), label))
            plt.close()
        log.info('Classification Report for model %s: \n%s', model.model_name, classification_report(y_true, y_pred))

    for training_config in confusion_matrix_paths:
        for path, caption, label in confusion_matrix_paths[training_config]:
            print('\\clearpage')
            print(f'\\begin{{figure}}[p]')
            print(f'\\centering')
            print(f'\\includegraphics[scale=0.5]{{images/{path}}}')
            print(f'\\caption[{caption}]{{{caption}}}\label{{fig:{label}}}')
            print(f'\\end{{figure}}')
            print()

if __name__ == '__main__':
    main()
