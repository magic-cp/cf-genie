import logger
logger.setup_applevel_logger(is_debug=True)

import utils

log = logger.get_logger(__name__)

log.info(utils.preprocess_cf_statement('This is a test case!'))
