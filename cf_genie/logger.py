import logging
import os
import sys
from logging import Logger
from pathlib import Path
from pprint import PrettyPrinter

APP_LOGGER_NAME = 'cf_genie'

LOGS_FOLDER = os.path.join(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))), 'logs')

Path(LOGS_FOLDER).mkdir(parents=True, exist_ok=True)


def setup_applevel_logger(logger_name=APP_LOGGER_NAME,
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
        fh = logging.FileHandler(os.path.join(
            LOGS_FOLDER, Path(file_name).stem) + '.logs')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(module_name):
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)


def get_prettyprinter():
    return PrettyPrinter(indent=4)


class Loggable:
    def __init__(self) -> None:
        klass = self.__class__
        module = klass.__module__

        # reference https://stackoverflow.com/questions/2020014/get-fully-qualified-class-name-of-an-object-in-python
        if module == 'builtins':
            log_name = klass.__qualname__  # avoid outputs like 'builtins.str'
        else:
            log_name = module + '.' + klass.__qualname__

        self.log = get_logger(log_name)
