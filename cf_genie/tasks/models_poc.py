import numpy as np
from sklearn.preprocessing import MinMaxScaler

import cf_genie.logger as logger
import cf_genie.utils as utils
from cf_genie.embedders import Doc2VecEmbedder
from cf_genie.models import ComplementNaiveBayes, MultinomialNaiveBayes
from cf_genie.utils.timer import Timer

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():
    df = utils.read_cleaned_dataset()

    with Timer('Reading embedded words using Doc2VecEmbedder', log=log):
        embedded_words = Doc2VecEmbedder.read_embedded_words()
    Y = df['most_occurrent_tag_group'].to_numpy()
    log.info('Y count %s', df['most_occurrent_tag_group'].value_counts())
    log.info('embedded words shape %s', embedded_words.shape)

    scaler = MinMaxScaler()
    scaler.fit(embedded_words, Y)
    # Training model with all data
    model = MultinomialNaiveBayes(scaler.transform(embedded_words), Y, version='imbalanced')

    log.info('Model name %s', model.model_name)
    preds = model.predict(embedded_words[:10])
    log.info('Sample predictions without balancing:')
    for tag, pred in zip(Y[:10], preds):
        log.info('Tag %s, Pred %s', tag, pred)
    log.info('MNB training score: %s', model.training_score())

    ###
    log.info('Training model with 50-50 data. ADHOC has most of the data, so let\'s split it')
    Y_adhoc = np.vectorize(lambda tag: 'ADHOC' if tag == 'ADHOC' else 'NON_ADHOC')(Y)
    model = MultinomialNaiveBayes(scaler.transform(embedded_words), Y_adhoc, version='balanced')
    preds = model.predict(embedded_words[:10])
    log.info('Sample predictions without balancing:')
    for tag, pred in zip(Y_adhoc[:10], preds):
        log.info('Tag %s, Pred %s', tag, pred)
    log.info('MNB training score: %s', model.training_score())

    ###
    log.info('Training model with CMB')
    model = ComplementNaiveBayes(scaler.transform(embedded_words), Y)
    preds = model.predict(embedded_words[:10])
    log.info('Sample predictions without balancing:')
    for tag, pred in zip(Y[:10], preds):
        log.info('Tag %s, Pred %s', tag, pred)
    log.info('CNB training score: %s', model.training_score())


if __name__ == '__main__':
    main()
