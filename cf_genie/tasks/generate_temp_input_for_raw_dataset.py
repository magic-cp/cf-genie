"""
Generate input csv with all codeforces problems
"""

import csv

import cf_genie.utils as utils


def main():
    with open(utils.PROBLEM_CONTEST_IDS_CSV, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=utils.CONTEST_PROBLEM_ID_FIELDNAMES)
        writer.writeheader()
        for problem in utils.load_problems():
            writer.writerow({
                utils.PROBLEM_ID: problem.index,
                utils.CONTEST_ID: problem.contest_id
            })


if __name__ == '__main__':
    main()
