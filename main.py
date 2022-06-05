import csv
import logger
logger.setup_applevel_logger(is_debug=True, file_name='app.logs')

from pprint import PrettyPrinter

pprint = PrettyPrinter().pprint


import utils

log = logger.get_logger(__name__)

with open('./dataset/cf_problems.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pprint(row['statement'])
        pprint(utils.preprocess_cf_statement(row['statement']))
