"""
Utilities to scrap the problem details of a codeforces problem.

Date created: 2022-05-16
Date of when codeforces was studied for scraping: 2022-05-16
"""
import json
import os
from dataclasses import dataclass
from typing import List

import cf_genie.utils.cf_api as cf_api
import cf_genie.utils.read_write_files as read_write_files

CF_PROBLEMS = os.path.join(read_write_files.TEMP_PATH, 'problems.json')
CF_CONTESTS = os.path.join(read_write_files.TEMP_PATH, 'contests.json')
PROBLEM_CONTEST_IDS_CSV = os.path.join(read_write_files.TEMP_PATH, 'problem_contest_ids.csv')


@dataclass
class Contest:
    """
    Small representation of a contest from CF API
    """
    name: str
    phase: str
    contest_id: int


@dataclass
class Problem:
    """
    Small representation of a problem from CF API
    """
    index: str
    contest_id: int
    name: str
    tags: List[str]
    solved_count: int = 0

    def get_url(self):
        return f'https://codeforces.com/contest/{self.contest_id}/problem/{self.index}'

    def __str__(self) -> str:
        return f'{self.contest_id}{self.index} - {self.name}'


def map_to_contest(cf_response):
    return [Contest(contest['name'], contest['phase'], contest['id']) for contest in cf_response['result']]


def map_to_problem(cf_response):
    problems = [Problem(problem['index'], problem['contestId'], problem['name'], problem['tags'])
                for problem in cf_response['result']['problems']]

    problem_to_solved_count = {}
    for stat in cf_response['result']['problemStatistics']:
        problem_to_solved_count[(stat['contestId'], stat['index'])] = stat['solvedCount']

    for problem in problems:
        problem.solved_count = problem_to_solved_count.get((problem.contest_id, problem.index), 0)
    return problems


def load_and_store_response(file_name, mapper):
    def dec(func):
        def wrapped(*args, **kwargs):
            force_reload = kwargs['force_reload'] if 'force_reload' in kwargs else False
            if not force_reload and os.path.exists(file_name):
                with open(file_name, 'r') as f:
                    return mapper(json.load(f))

            problems_json = func()

            with open(file_name, 'w') as f:
                json.dump(problems_json, f, indent=2)

            return mapper(problems_json)
        return wrapped
    return dec


@load_and_store_response(CF_PROBLEMS, map_to_problem)
def load_problems() -> List[Problem]:
    return cf_api.get_problems()


@load_and_store_response(CF_CONTESTS, map_to_contest)
def load_contests() -> List[Contest]:
    return cf_api.get_contests()
