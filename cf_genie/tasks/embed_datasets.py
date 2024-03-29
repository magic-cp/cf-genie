import numpy as np
from tqdm import tqdm

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.embedders import EMBEDDERS

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)

tqdm.pandas()


def main():

    # train embedders
    EMBEDDERS_CONFIG = [
        # balanced
        {
            'all_data_id': 'without-adhoc-train-balanced',
            'datasets_to_embed': ['without-adhoc-test', 'without-adhoc-train-balanced']
        },
        # with adhoc
        {
            'all_data_id': 'train',
            'datasets_to_embed': ['test', 'train']
        }
    ]
    for embedder_config in EMBEDDERS_CONFIG:
        all_data_id = embedder_config['all_data_id']
        df = utils.read_cleaned_dataset(all_data_id)
        for embedder_class in EMBEDDERS:
            with utils.Timer(f'Training embedder {embedder_class.__name__}'):
                embedder = embedder_class(df['preprocessed_statement'].to_numpy(), training_label=all_data_id)

        for label in embedder_config['datasets_to_embed']:
            df = utils.read_cleaned_dataset(label)
            for embedder_class in EMBEDDERS:
                embedder = embedder_class([], label=label, training_label=all_data_id)

                with utils.Timer(f'Embedding with {embedder.embedder_name}', log=log):
                    words = np.array(df['preprocessed_statement'].progress_apply(lambda x: embedder.embed(x)).tolist())
                with utils.Timer(f'Writing words of {embedder.embedder_name_with_label} to file', log=log):
                    embedder.write_embedded_words(words)


if __name__ == '__main__':
    main()
