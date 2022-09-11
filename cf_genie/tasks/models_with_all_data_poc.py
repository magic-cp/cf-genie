
from itertools import product

import numpy as np
from sklearn import model_selection

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.embedders import EMBEDDERS
from cf_genie.models import SUPERVISED_MODELS, TrainingMethod
from cf_genie.models.base import BaseModel
from cf_genie.models.model_runner import RunStrategy, removing_one_class, run_model
from cf_genie.utils.timer import Timer

logger.setup_applevel_logger(
    is_debug=True, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():
    df = utils.read_cleaned_dataset('without-adhoc-train')
    y = df['most_occurrent_tag_group'].to_numpy()


    for model_class, embedder_class in product(SUPERVISED_MODELS, EMBEDDERS):
        embedder = embedder_class([], label='without-adhoc-train')
        removing_one_class(model_class, embedder, y, tag_group='ADHOC')


if __name__ == '__main__':
    main()
