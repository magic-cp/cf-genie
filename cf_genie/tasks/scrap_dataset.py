"""
Scrapes the entire CF problemset and writes the dataset to a csv file.
"""
import argparse
import csv
import os
import traceback
from typing import Optional

import requests
from bs4 import BeautifulSoup

import cf_genie.logger as logger
import cf_genie.utils as utils

PROBLEM_STATEMENT_SELECTOR = '.problem-statement'
TITLE_SELECTOR = PROBLEM_STATEMENT_SELECTOR + ' .title'
STATEMENT_SELECTOR = PROBLEM_STATEMENT_SELECTOR + ' .header + div > p'
INPUT_SPECIFICATION_SELECTOR = PROBLEM_STATEMENT_SELECTOR + ' .input-specification > p'
OUTPUT_SPECIFICATION_SELECTOR = PROBLEM_STATEMENT_SELECTOR + ' .output-specification > p'
TAG_SELECTOR = 'span.tag-box'


logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=False)

log = logger.get_logger(__name__)

DATASET_FILE = os.path.join(utils.TEMP_PATH, 'raw_dataset_file.csv')


def get_csv_reader(file_name):
    with open(file_name, 'w+') as csv_file:
        return csv.DictReader(csv_file)


def get_csv_writer(file_name):
    with open(file_name, 'w+') as csv_file:
        return csv.DictWriter(csv_file, fieldnames=utils.RAW_CSV_FIELDNAMES)


def get_problem_title(soup: BeautifulSoup):
    # we splice because the problem title has the problem id in it e.g. "A. Bit++"
    return soup.select_one(TITLE_SELECTOR).text[3:]


def get_problem_statement(soup: BeautifulSoup):
    s = []
    for p in soup.select(STATEMENT_SELECTOR):
        s.append(p.get_text())
    return ' '.join(s)


def get_input_spec(soup: BeautifulSoup):
    s = []
    for p in soup.select(INPUT_SPECIFICATION_SELECTOR):
        s.append(p.get_text())
    return ' '.join(s)


def get_output_spec(soup: BeautifulSoup) -> Optional[str]:
    s = []
    for p in soup.select(OUTPUT_SPECIFICATION_SELECTOR):
        s.append(p.get_text())
    if s:
        return ' '.join(s)
    return None


def get_tags(soup: BeautifulSoup):
    tags = []
    for tag in soup.select(TAG_SELECTOR):
        tags.append(tag.text.strip())
    return ';'.join(tags)


def get_problem_details(contest_id, problem_id, rcpc):

    url = f"https://codeforces.com/contest/{contest_id}/problem/{problem_id}"

    log.debug(url)
    cookies = {
        'RCPC': rcpc,
    }
    page = requests.get(url, cookies=cookies)

    soup = BeautifulSoup(page.content, "html.parser")

    # Might be None if it's an interactive problem
    output_spec = get_output_spec(soup)

    return {
        utils.CONTEST_ID: contest_id,
        utils.PROBLEM_ID: problem_id,
        utils.TITLE: get_problem_title(soup),
        utils.STATEMENT: get_problem_statement(soup),
        utils.INPUT_SPEC: get_input_spec(soup),
        utils.OUTPUT_SPEC: output_spec if output_spec else '',
        utils.URL_KEY: url,
        utils.TAGS: get_tags(soup),
        utils.IS_INTERACTIVE: output_spec is None
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rcpc', required=True, help='RCPC cookie', type=str)
    return parser.parse_args()


def main():
    existing_problem_ids = set()
    try:
        with open(DATASET_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_problem_ids.add((row[utils.CONTEST_ID], row[utils.PROBLEM_ID]))
    except FileNotFoundError:
        print('Dataset file does not exist. Creating it')

    args = parse_args()

    log.debug('RCPC token: %s', args.rcpc)
    log.info('Resulting dataset file: %s', DATASET_FILE)

    with open(DATASET_FILE, 'a+', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=utils.RAW_CSV_FIELDNAMES)
        if len(existing_problem_ids) == 0:
            writer.writeheader()

        with open(utils.PROBLEM_CONTEST_IDS_CSV, 'r+') as f:
            reader = csv.DictReader(f)
            for input_row in reader:
                if (input_row[utils.CONTEST_ID], input_row[utils.PROBLEM_ID]) not in existing_problem_ids:
                    log.info('Processing: %s%s', input_row[utils.CONTEST_ID], input_row[utils.PROBLEM_ID])
                    try:
                        problem_details = get_problem_details(
                            input_row[utils.CONTEST_ID], input_row[utils.PROBLEM_ID], args.rcpc)
                        # log.info('Problem info', problem_details)
                        log.info(f'Problem %s%s processed', input_row[utils.CONTEST_ID], input_row[utils.PROBLEM_ID])
                        writer.writerow(problem_details)
                    except Exception as e:
                        log.warn('Failed to process: %s%s', input_row[utils.CONTEST_ID], input_row[utils.PROBLEM_ID])
                        log.warn(e)
                        log.warn(traceback.format_exc())
                else:
                    log.debug('Problem already processed: %s%s',
                              input_row[utils.CONTEST_ID], input_row[utils.PROBLEM_ID])


if __name__ == '__main__':
    main()
