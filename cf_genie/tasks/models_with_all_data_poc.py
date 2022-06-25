
import numpy as np

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.embedders import EMBEDDERS
from cf_genie.models import SUPERVISED_MODELS
from cf_genie.models.base import BaseModel
from cf_genie.utils.timer import Timer

logger.setup_applevel_logger(
    is_debug=True, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def print_train_results(model: BaseModel, X, y):
    log.info('Model name %s', model.model_name)
    preds = model.predict(X[:10])
    log.info('Sample predictions on model %s:', model.model_name)
    for tag, pred in zip(y[:10], preds):
        log.info('Tag %s, Pred %s', tag, pred)
    # log.info('%s score: %s', model.model_name, model.training_score())


def main():
    df = utils.read_cleaned_dataset()
    y = df['most_occurrent_tag_group'].to_numpy()
    with Timer('Running task ' + __file__, log=log):
        for embedder_class in EMBEDDERS:
            for model_class in SUPERVISED_MODELS:
                with Timer(f'Loading embedded words for {embedder_class.__name__}', log=log):
                    X = embedder_class.read_embedded_words()
                with Timer(f'Training model {model_class.__name__} with embedder {embedder_class.__name__}', log=log):
                    model = model_class(embedder_class.read_embedded_words, y, label='with-' + embedder_class.__name__ + '-on-imbalanced-data')
                    print_train_results(model, X, y)


if __name__ == '__main__':
    main()
