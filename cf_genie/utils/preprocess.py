import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

import cf_genie.logger as logger

nltk.download('punkt')
nltk.download('stopwords')

stemmer = SnowballStemmer(language='english')

log = logger.get_logger(__name__)

STOPWORDS = set(stopwords.words('english')).union(
    {'help', 'helping', 'helped', 'helps', 'want', 'wants', 'wanted'})

CONTRACTIONS = {
    "n't": 'not',
    "'s": 'is',  # may be "has", but "is" is more general, and both option are stopwords hence will be remove
    "'ve": 'have',
    "'ll": 'will',
    "'d": 'had',
}

CONTRACTIONS_PREVIOUS_ADJUSTMENTS = {
    'ca': 'can',  # can't
    'wo': 'will',  # won't
}

# This expression matches the following mathematical expressions:
# term<op>term
#   where term is either a number or a variable
#   and op is either +, -
VARIABLE_REGEX = r'[a-z](_[0-9]+)?'
NUMBERS_AND_VARIABLES = r'([0-9]+|' + VARIABLE_REGEX + ')'
MATHEMATICAL_OPERATORS = r'[+-]'
MATHEMATICAL_EXPRESSION_REGEX = rf'^({NUMBERS_AND_VARIABLES})({MATHEMATICAL_OPERATORS}{NUMBERS_AND_VARIABLES})*$'

LATEX_TOKENS = {
    r'\dots',
    r'\dot',
    r'\le',
    r'\cdot',
    r'\times',
    r'\ldot',
    r'\gcd',
    r'\bmod',
    r'\ge',
    r'\text',
    r'\leq',
    r'\ldot',
    r'\ldots',
    r'\texttt',
    r'\begin',
    r'\sum\limits_',
    r'\in',
    r'\to',
    r'\neq',
    r'\frac',
    r'\operatornam',
    r'\sum_',
    r'\oplus',
    r'\max\limits_',
    r'\alpha',
    r'\ne',
    r'\geq',
    r'\gt',
    r'\sigma',
}


def handle_contractions(tokens: List[str]):
    # Handle contractions
    tokens_without_contractions = ['' for _ in tokens]
    for i in range(0, len(tokens)):
        if tokens[i] in CONTRACTIONS:
            tokens_without_contractions[i] = CONTRACTIONS[tokens[i]]

            # We need to adjust the previous token. For example, `can't` is
            # tokenized to `ca` and `n't`, so the previous `ca` should be `can`
            if i > 0 and tokens[i - 1] in CONTRACTIONS_PREVIOUS_ADJUSTMENTS:
                tokens_without_contractions[i - 1] = CONTRACTIONS_PREVIOUS_ADJUSTMENTS[tokens[i - 1]]

            # "let's" should be relaxed to "let us", not "let is". Neither "let" or "us" are stopwords
            if i > 0 and tokens[i - 1] == 'let':
                tokens_without_contractions[i] = 'us'
        else:
            tokens_without_contractions[i] = tokens[i]

    return tokens_without_contractions


def remove_punctuation(text: List[str]) -> List[str]:
    def has_only_punctuation(x): return all(c in string.punctuation for c in x)
    return [word for word in text if not has_only_punctuation(word)]


def remove_stopwords(text: List[str]) -> List[str]:
    return [word for word in text if word not in STOPWORDS]


def stem_words(words: List[str]) -> List[str]:
    return [stemmer.stem(word) for word in words]


def remove_mathematical_expressions(tokens: List[str]) -> List[str]:
    # TODO: This might not be enough to handle all posible math expressions. Ideally, we should try to use
    #       pylatexenc.readthedocs.io to properly handle latex notation.
    return [token for token in tokens if not re.match(
        MATHEMATICAL_EXPRESSION_REGEX, token)]


def remove_simple_latex_tokens(tokens: List[str]) -> List[str]:
    return [token for token in tokens if token not in LATEX_TOKENS]


def preprocess_cf_statement(text: str) -> List[str]:
    log.debug('Text input: %s', text)

    # Lowercase the entire text
    text = text.lower()
    log.debug('Text after lowercase: %s', text)

    # Word tokenization
    words = word_tokenize(text)
    log.debug('Tokens: %s', words)

    # Punctuation
    words = remove_punctuation(words)
    log.debug('Text after punctuation: %s', words)

    # Contraction re-mapping
    words = handle_contractions(words)
    log.debug('Tokens after contractions: %s', words)

    # Stopword removal
    words = remove_stopwords(words)
    log.debug('Tokens after stopwords: %s', words)

    # Mathematical expression removal
    # TODO: Circle back to this preprocessing step, as it may add value to the
    # model to know about mathematical expressions
    words = remove_mathematical_expressions(words)
    log.debug('Tokens after mathematical expressions: %s', words)

    words = remove_simple_latex_tokens(words)
    log.debug('Tokens after simple latex tokens: %s', words)

    # Stemmization
    words = stem_words(words)
    log.debug('Tokens after stemming: %s', words)

    return words


def preprocess_and_store(dataset: List[str]):
    for i in range(0, len(dataset)):
        dataset[i] = preprocess_cf_statement(dataset[i])
