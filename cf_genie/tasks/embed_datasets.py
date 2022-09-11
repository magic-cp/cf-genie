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

    for label in ['without-adhoc-train', 'without-adhoc-test']:
        if label:
            df = utils.read_cleaned_dataset(label)
        else:
            df = utils.read_cleaned_dataset()
        for embedder_class in EMBEDDERS:
            if label:
                embedder = embedder_class(df['preprocessed_statement'].to_numpy(), label)
            else:
                embedder = embedder_class(df['preprocessed_statement'].to_numpy())

            with utils.Timer(f'Embedding with {embedder.embedder_name}', log=log):
                words = np.array(df['preprocessed_statement'].progress_apply(lambda x: embedder.embed(x)).tolist())
            with utils.Timer(f'Writing words of {embedder.embedder_name} to file', log=log):
                embedder.write_embedded_words(words)


if __name__ == '__main__':
    main()
