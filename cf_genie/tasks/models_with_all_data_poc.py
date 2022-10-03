
from itertools import product

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.embedders import EMBEDDERS
from cf_genie.models import SUPERVISED_MODELS
from cf_genie.models.model_runner import all_strategy

logger.setup_applevel_logger(
    is_debug=True, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)

TRAINING_DATASET = [
    # 'without-adhoc-train-balanced',
    'train'
]

def main():
    for training_dataset in TRAINING_DATASET:
        df = utils.read_cleaned_dataset(training_dataset)
        y = df['most_occurrent_tag_group'].to_numpy()

        for model_class, embedder_class in product(SUPERVISED_MODELS, EMBEDDERS):
            embedder = embedder_class([], label=training_dataset, training_label=training_dataset)
            all_strategy(model_class, embedder, y)


if __name__ == '__main__':
    main()
