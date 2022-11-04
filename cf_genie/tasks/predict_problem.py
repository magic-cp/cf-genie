from itertools import product

import argparse

import cf_genie.logger as logger
from cf_genie import utils
from cf_genie.embedders import FastTextEmbedder150
from cf_genie.models import MLP
from cf_genie.models.model_runner import get_model_suffix_name_for_all_classes

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)

def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Splits the dataset into training and test split')
    parser.add_argument(
        '--contest-id',
        type=int,
        help='Contest id',
        required=True)
    parser.add_argument(
        '--problem-idx',
        type=str,
        help='Problem index',
        required=True)
    if args:
        return parser.parse_args(args)
    else:
        return parser.parse_args()


def main(*args):
    args = parse_args(args)
    contest_id = args.contest_id
    problem_idx = args.problem_idx

    df_all = utils.read_cleaned_dataset()
    problems_df = df_all[(df_all['contest_id'] == contest_id) & (df_all['problem_id'] == problem_idx)]

    training_dataset = 'without-adhoc-train-balanced'
    y_true_train = utils.read_cleaned_dataset(training_dataset)['most_occurrent_tag_group'].to_numpy()

    model_class = MLP
    embedder_class = FastTextEmbedder150
    embedder = embedder_class([], label=training_dataset, training_label=training_dataset)
    model = model_class(
        embedder.read_embedded_words,
        y_true_train,
        label=get_model_suffix_name_for_all_classes(embedder.embedder_name))

    X = problems_df['preprocessed_statement'].apply(lambda x: embedder.embed(x)).tolist()
    XX = problems_df[['contest_id',
                        'problem_id',
                        'most_occurrent_tag_group',
                        'tag_groups',
                        'cleaned_tags']].values.tolist()
    y = model.predict(X)
    for problem_id, prediction in zip(XX, y):
        print('Prediction', prediction, sep='\t')
        print('Actual', problem_id[2], sep='\t')
        print('Tags', problem_id[4])


if __name__ == '__main__':
    main()
