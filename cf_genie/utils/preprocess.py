import re
import string
from enum import Enum, auto
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

import cf_genie.logger as logger

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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

# TODO: Review these. It might be useful to re-map these to something else.
# TODO: This is not currently used, but it might be useful in the future.
LATEX_TOKENS = {
    r'\cdot',
    r'\cdots',
    r'\equiv',
    r'\quad',
    r'\textrm',
    r'\sum_',
    r'\in',
    r'\subseteq',
    r'\max',
    r'\max_',
    r'\prod_',
    r'\ldot',
    r'\ldots',
    r'\geq',
    r'\dot',
    r'\dots',
    r'\texttt',
    r'\bmod',
    r'\left',
    r'\right',
    r'\large',
    r'\operatorname',
    r'\substack',
    r'\le',
    r'\gcd',
    r'\min\limits_',
    r'\min_',
    r'\min',
    r'\sum\limits_',
    r'\to',
    r'\langle',
    r'\rangle',
    r'\oplus',
    r'\frac',
    r'\leq',
    r'\underline',
    r'\land',
    r'\cdots\land',
    r'\sigma_d',
    r'\smash',
    r'\displaystyle\max_',
    r'\log',
    r'\delta^',
    r'\vec',
    r'\mathbb',
    r'\neq',
    r'\varnothing',
    r'\left\\',
    r'\text\\'}


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

    def remove_punctuation(x: str):
        # handling latex tokens
        if x[0] == '\\':
            return x
        return x.translate(str.maketrans('', '', string.punctuation))
    return [remove_punctuation(word) for word in text if not has_only_punctuation(word)]


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
    # The following code is sort of naive. It simply removes all tokens that start with a backslash.
    return [token for token in tokens if not token.startswith('\\')]


def remove_latex_begin_end_blocks(text: str) -> str:
    ret = []
    i = 0
    l = len(text)
    beg = '\\begin'
    end = '\\end'
    while i < l - 6:
        if text[i:i + 6] == beg:
            num_begins = 1
            i += 6
            while num_begins and i < l - 4:
                if text[i:i + 6] == beg:
                    num_begins += 1
                    i += 6
                elif text[i:i + 4] == end:
                    num_begins -= 1
                    i += 4
                else:
                    i += 1
        else:
            ret.append(text[i])
            i += 1
    while i < l:
        ret.append(text[i])
        i += 1
    return ''.join(ret)


def remove_latex_dollar_blocks(text: str) -> str:
    ret = []
    i = 0
    l = len(text)
    while i < l - 3:
        if text[i:i + 3] == '$$$':
            i += 3
            while i < l - 3 and text[i:i + 3] != '$$$':
                i += 1
            i += 3
        else:
            ret.append(text[i])
            i += 1
    while i < l:
        ret.append(text[i])
        i += 1
    return ''.join(ret)


class LatexTokenAction(Enum):
    REMOVE = auto()
    KEEP_AND_REMAP = auto()


def preprocess_cf_statement(text: str, latext_token_action: LatexTokenAction = LatexTokenAction.REMOVE) -> List[str]:
    log.debug('Text input: %s', text)

    # Lowercase the entire text
    text = text.lower()
    log.debug('Text after lowercase: %s', text)

    # Latex \begin and \end removal
    text = remove_latex_begin_end_blocks(text)
    log.debug('Text after latex begin end removal: %s', text)

    # Latex $$$ block removal
    text = remove_latex_dollar_blocks(text)
    log.debug('Text after latex block removal: %s', text)

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
