import logging
import sys
from pprint import PrettyPrinter

APP_LOGGER_NAME = 'cf_topic_modeling'

def setup_applevel_logger(logger_name = APP_LOGGER_NAME,
                        is_debug=True,
                        file_name=None,
                        simple_logs=True):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG if is_debug else logging.INFO)

    if simple_logs:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    else:
        formatter = logging.Formatter('[%(levelname)s] %(message)s')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)

    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(module_name):
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)

def get_prettyprinter():
    return PrettyPrinter(indent=4)
