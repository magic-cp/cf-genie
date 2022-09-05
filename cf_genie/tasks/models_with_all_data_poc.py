
from itertools import product

import numpy as np
from sklearn import model_selection

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.embedders import EMBEDDERS
from cf_genie.models import SUPERVISED_MODELS, TrainingMethod
from cf_genie.models.base import BaseModel
from cf_genie.models.model_runner import RunStrategy, run_model
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
    for run_strategy, model_class in product(RunStrategy, SUPERVISED_MODELS):
        run_model(model_class, y, run_strategy)


if __name__ == '__main__':
    main()
