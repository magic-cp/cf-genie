from itertools import product

import argparse

import subprocess
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
    parser.add_argument(
        '--problem-parser',
        type=str,
        help='Problem parser',
        required=True)
    parser.add_argument(
        '--with-input-constants',
        action='store_true'
    )
    parser.add_argument(
        '--with-output-constants',
        action='store_true'
    )
    parser.add_argument(
        '--with-test-cases',
        action='store_true'
    )
    if args:
        return parser.parse_args(args)
    else:
        return parser.parse_args()


def main(*args):
    args = parse_args(args)
    contest_id = args.contest_id
    problem_idx = args.problem_idx
    problem_parser = args.problem_parser

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
    prediction = y[0]
    log.info('Invoking magic-cp:')


    magic_cp_cmd = [
        "magic-cp",
        "--config", "magic-cp-config.json",
        "run-with-problem-category",
        "--cf-problem-idx", problem_idx,
        "--cf-contest-id", str(contest_id),
        "--problem-category", prediction,
        "--problem-parser", problem_parser]

    if args.with_input_constants:
        magic_cp_cmd.append("--with-input-constants")
    
    if args.with_output_constants:
        magic_cp_cmd.append("--with-output-constants")

    if args.with_test_cases:
        magic_cp_cmd.append("--with-test-cases")

    log.info('Invoking magic-cp with command: %s', magic_cp_cmd)
    subprocess.run(magic_cp_cmd)



if __name__ == '__main__':
    main()
