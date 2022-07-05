
import numpy as np

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.embedders import EMBEDDERS
from cf_genie.models import SUPERVISED_MODELS, TrainingMethod
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
                with Timer(f'Training model {model_class.__name__} with embedder {embedder_class.__name__} on all classes', log=log):
                    model = model_class(
                        embedder_class.read_embedded_words,
                        y,
                        TrainingMethod.GRID_SEARCH_CV,
                        label='with-' +
                        embedder_class.__name__ +
                        '-on-all-classes-grid-search-cv')
                    print_train_results(model, X, y)
                for tag_group in utils.TAG_GROUPS:
                    non_tag_group = f'NON_{tag_group}'
                    y_tag_group = np.vectorize(lambda x: tag_group if x == tag_group else non_tag_group)(y)
                    log.info(np.unique(y_tag_group))
                    with Timer(f'Training model {model_class.__name__} with embedder {embedder_class.__name__} on tag group {tag_group} vs others', log=log):
                        model = model_class(
                            embedder_class.read_embedded_words,
                            y_tag_group,
                            label=f'with-{embedder_class.__name__}-on-{tag_group}-vs-rest-classes')
                        print_train_results(model, X, y_tag_group)

                    with Timer(f'Training model {model_class.__name__} with embedder {embedder_class.__name__} on all classes except {tag_group} data', log=log):
                        y_not_tag_group = y != tag_group

                        def get_x():
                            X = embedder_class.read_embedded_words()
                            return X[y_not_tag_group]
                        model = model_class(
                            get_x,
                            y[y_not_tag_group],
                            label=f'with-{embedder_class.__name__}-without-{tag_group}-class')
                        print_train_results(model, get_x(), y[y_not_tag_group])


if __name__ == '__main__':
    main()
