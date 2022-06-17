from typing import List, Tuple, Type

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.embedders import EMBEDDERS, Doc2VecEmbedder
from cf_genie.models import (SUPERVISED_MODELS, ComplementNaiveBayes,
                             MultinomialNaiveBayes)
from cf_genie.models.base import BaseModel, BaseSupervisedModel
from cf_genie.utils.timer import Timer

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)

def print_train_results(model: BaseModel, X_train, y_train, X_test, y_test):
    log.info('Model name %s', model.model_name)
    preds = model.predict(X_train[:10])
    log.info('Sample predictions on training data on model %s:', model.model_name)
    for tag, pred in zip(y_train[:10], preds):
        log.info('Tag %s, Pred %s', tag, pred)
    log.info('%s raining score: %s', model.model_name, model.training_score())

    preds = model.predict(X_test[:10])
    log.info('Sample predictions on training data %s:', model.model_name)
    for tag, pred in zip(y_test[:10], preds):
        log.info('Tag %s, Pred %s', tag, pred)
    log.info('%s test score: %s', model.model_name, model.test_score(X_test, y_test))


def main():
    df = utils.read_cleaned_dataset()
    # for embedder_class in EMBEDDERS:
    #     embeder_name = embedder_class.__name__
    #     with Timer('Reading words using {}'.format(embeder_name)):
    #         words = embedder_class.read_embedded_words()

    #     Y = df['most_occurrent_tag_group'].to_numpy()
    #     log.info('Y count %s', df['most_occurrent_tag_group'].value_counts())
    #     log.info('embedded words shape %s', words.shape)

    #     scaler = MinMaxScaler()
    #     with Timer('Fitting MinMaxScaler', log=log):
    #         scaler.fit(words, Y)
    #     # Training model with all data
    #     with Timer('Building MNB with imbalance data, no splitting', log=log):
    #         model = MultinomialNaiveBayes(scaler.transform(words), Y, run_in_mongo=embedder_class.USE_IN_MONGODB_HYPEROPT, version=f'imbalanced-using-{embeder_name}')

    #     log.info('Model name %s', model.model_name)
    #     preds = model.predict(words[:10])
    #     log.info('Sample predictions without balancing:')
    #     for tag, pred in zip(Y[:10], preds):
    #         log.info('Tag %s, Pred %s', tag, pred)
    #     log.info('MNB training score: %s', model.training_score())

    #     ###
    #     log.info('Training model with 50-50 data. ADHOC has most of the data, so let\'s split it')
    #     Y_adhoc = np.vectorize(lambda tag: 'ADHOC' if tag == 'ADHOC' else 'NON_ADHOC')(Y)
    #     model = MultinomialNaiveBayes(scaler.transform(words), Y_adhoc, run_in_mongo=embedder_class.USE_IN_MONGODB_HYPEROPT, version=f'balanced-using-{embeder_name}')
    #     preds = model.predict(words[:10])
    #     log.info('Sample predictions without balancing:')
    #     for tag, pred in zip(Y_adhoc[:10], preds):
    #         log.info('Tag %s, Pred %s', tag, pred)
    #     log.info('MNB training score: %s', model.training_score())

    #     ###
    #     log.info('Training model with CMB')
    #     model = ComplementNaiveBayes(scaler.transform(words), Y, run_in_mongo=embedder_class.USE_IN_MONGODB_HYPEROPT, version=f'using-{embeder_name}')
    #     preds = model.predict(words[:10])
    #     log.info('Sample predictions without balancing:')
    #     for tag, pred in zip(Y[:10], preds):
    #         log.info('Tag %s, Pred %s', tag, pred)
    #     log.info('CNB training score: %s', model.training_score())

    log.info("Let's test now with splitting data")
    SPLIT_PERCENTAGES: List[Tuple[str, float]] = [('ten-percent', 0.1), ('twenty-percent', 0.2), ('thirty-percent', 0.3), ('fourty-percent', 0.4), ('fifty-percent', 0.5)]
    for embedder_class in EMBEDDERS:
        embeder_name = embedder_class.__name__
        with Timer('Reading words using {}'.format(embeder_name)):
            words = embedder_class.read_embedded_words()

        Y = df['most_occurrent_tag_group'].to_numpy()
        log.info('Y count %s', df['most_occurrent_tag_group'].value_counts())
        log.info('embedded words shape %s', words.shape)

        for percentage_label, percentage in SPLIT_PERCENTAGES:
            # stratify makes sure that we keep the imbalance
            X_train, X_test, y_train, y_test = train_test_split(
                words, Y, test_size=percentage, random_state=42, stratify=Y)

            scaler = MinMaxScaler()
            scaler.fit(X_train, y_train)
            # Training model with all data
            model = MultinomialNaiveBayes(scaler.transform(X_train), y_train, version=f'training-data-{percentage_label}-imbalanced-using-{embeder_name}')

            print_train_results(model, X_train, y_train, X_test, y_test)

            ###
            log.info('Training model with 50-50 data. ADHOC has most of the data, so let\'s split it')

            # stratify makes sure that we keep the imbalance
            Y_adhoc = np.vectorize(lambda tag: 'ADHOC' if tag == 'ADHOC' else 'NON_ADHOC')(Y)
            X_train, X_test, y_train, y_test = train_test_split(
                words, Y_adhoc, test_size=percentage, random_state=42, stratify=Y_adhoc)

            scaler = MinMaxScaler()
            scaler.fit(X_train, y_train)

            model = MultinomialNaiveBayes(scaler.transform(X_train), y_train, version=f'training-data-{percentage_label}-balanced-using-{embeder_name}')


            print_train_results(model, X_train, y_train, X_test, y_test)

            ###
            X_train, X_test, y_train, y_test = train_test_split(
                words, Y, test_size=percentage, random_state=42, stratify=Y)

            scaler = MinMaxScaler()
            scaler.fit(X_train, y_train)

            model = ComplementNaiveBayes(scaler.transform(X_train), y_train, version=f'training-data-{percentage_label}-using-{embeder_name}')


            print_train_results(model, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
