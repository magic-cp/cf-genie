import csv

import logger

logger.setup_applevel_logger(is_debug=False, file_name='app.logs', simple_logs=False)

from pprint import PrettyPrinter

pprint = PrettyPrinter().pprint


import utils

log = logger.get_logger(__name__)

with open('./dataset/cf_problems.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # pprint(row['statement'])
        res = utils.preprocess_cf_statement(row['statement'])
        # pprint(res)
        log.info('%s %s', row['url'],  res)
