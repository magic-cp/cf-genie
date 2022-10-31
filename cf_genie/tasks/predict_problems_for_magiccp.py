from itertools import product

from sklearn.metrics import f1_score, hamming_loss

import cf_genie.logger as logger
from cf_genie import utils
from cf_genie.embedders import *
from cf_genie.models import *
from cf_genie.models.model_runner import get_model_suffix_name_for_all_classes

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():

    REPORT_CONFIG = [
        # Uncomment to test the imbalanced case
        {
            'test_dataset': 'without-adhoc-test',
            'training_dataset': 'without-adhoc-train-balanced',
            'result_label': r'balanceado sin \adhoc{}',
        }
    ]
    results = {r'con \adhoc': {}, r'balanceado sin \adhoc{}': {}}

    def to_percentage(x): return f'{x * 100:.0f}\\%'
    PROBLEMS_MAGICCP = [
        (115, 'A'),
        (939, 'A'),
        (292, 'B'),
        (500, 'A'),
        (893, 'C'),
        (522, 'A'),
        (1030, 'B'),
        (514, 'B'),
        (1642, 'A'),
        (993, 'A'),
        (127, 'A'),
        (1047, 'B'),
        (486, 'A'),
        (148, 'A'),
        (1328, 'A'),
        (581, 'A'),
        (630, 'C'),
        (1514, 'B'),
    ]

    df_all = utils.read_cleaned_dataset()
    df_lookup = list(map(lambda x: (df_all['contest_id'] == x[0]) & (df_all['problem_id'] == x[1]), PROBLEMS_MAGICCP))

    cond = df_lookup[0]
    for c in df_lookup[1:]:
        cond = cond | c

    problems_df = df_all[cond]

    training_dataset = 'without-adhoc-train-balanced'
    y_true_train = utils.read_cleaned_dataset(training_dataset)['most_occurrent_tag_group'].to_numpy()

    for model_class, embedder_class in product([MLP], [FastTextEmbedder100, FastTextEmbedder150]):
        embedder = embedder_class([], label=training_dataset, training_label=training_dataset)
        model = model_class(
            embedder.read_embedded_words,
            y_true_train,
            label=get_model_suffix_name_for_all_classes(embedder.embedder_name))

        X = problems_df['preprocessed_statement'].apply(lambda x: embedder.embed(x)).tolist()
        log.info('Predicting with model %s', model.model_name)
        XX = problems_df[['contest_id',
                          'problem_id',
                          'most_occurrent_tag_group',
                          'tag_groups',
                          'cleaned_tags']].values.tolist()
        y = (model.predict(X))
        for problem_id, prediction in zip(XX, y):
            print(str(problem_id[0]) + problem_id[1], problem_id[2], prediction, '|', problem_id[3], "|", problem_id[4])


if __name__ == '__main__':
    main()
