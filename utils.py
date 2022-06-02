import logger
import string

log = logger.get_logger(__name__)

def preprocess_cf_statement(text: str) -> str:
    log.debug('Text input: %s', text)

    # Lowercase the entire text
    text = text.lower()
    log.debug('Text after lowercase: %s', text)

    # Punctuation
    text = ''.join(char for char in text if char not in string.punctuation)
    log.debug('Text after punctuation: %s', text)

    return text
