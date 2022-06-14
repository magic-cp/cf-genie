import numpy as np
from tqdm import tqdm

import cf_genie.utils as utils
from cf_genie.embedders import EMBEDDERS

tqdm.pandas()


def main():
    df = utils.read_cleaned_dataset()

    for embedder_class in EMBEDDERS:
        embedder = embedder_class(df['preprocessed_statement'].to_numpy())
        with utils.Timer(f'Embedding with {embedder.embedder_name}'):
            words = np.array(df['preprocessed_statement'].progress_apply(lambda x: embedder.embed(x)).tolist())
        with utils.Timer(f'Writing words of {embedder.embedder_name} to file'):
            embedder.write_embedded_words(words)
    pass


if __name__ == '__main__':
    main()
