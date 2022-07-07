import argparse
import importlib

# isort: off
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # disable tensorflow logging
# isort: on

from types import ModuleType

import cf_genie.logger as logger
from cf_genie.utils import Timer


def get_task_module(task):
    return f'cf_genie.tasks.{task}'


def import_module(task: str) -> ModuleType:
    try:
        return importlib.import_module(get_task_module(task))
    except ModuleNotFoundError as e:
        e.msg = f'Task {task} is not present in the cf_genie.tasks module'
        raise e


def main():
    parser = argparse.ArgumentParser(description='Run a task inside cf_genie.tasks')
    parser.add_argument('task', type=import_module, help='Python module inside cf_genie.tasks')

    args = parser.parse_args()
    module: ModuleType = args.task
    module.__name__

    log = logger.get_logger('task_runner')
    with Timer(f'Executing `main` function of module {module.__name__}', log=log):
        module.main()


if __name__ == '__main__':
    main()
