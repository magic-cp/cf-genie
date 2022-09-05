import argparse
import importlib
from types import ModuleType
from typing import Tuple

import cf_genie.logger as logger
from cf_genie.utils import Timer

# isort: off
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable tensorflow logging
# isort: on


def get_task_module(task):
    return f'cf_genie.tasks.{task}'


def import_module(task: str) -> ModuleType:
    try:
        return importlib.import_module(get_task_module(task))
    except ModuleNotFoundError as e:
        e.msg = f'Task {task} is not present in the cf_genie.tasks module'
        raise e

def parse_prog_arg(prog_arg: str) -> Tuple[str, str]:
    prog_arg_split = prog_arg.split('=')
    if len(prog_arg_split) != 2:
        raise ValueError(f"Found more than one '=' in program argument {prog_arg}")

    return prog_arg_split[0], prog_arg_split[1]

def main():
    parser = argparse.ArgumentParser(description='Run a task inside cf_genie.tasks')
    parser.add_argument('task', type=import_module, help='Python module inside cf_genie.tasks')
    parser.add_argument('--prog-arg', type=parse_prog_arg, help='Task runner arguments in the form of ARG=VALUE', action='append', default=[])

    args = parser.parse_args()

    print(args)
    module: ModuleType = args.task
    module.__name__

    log = logger.get_logger('task_runner')
    with Timer(f'Executing `main` function of module {module.__name__}', log=log):
        module.main()


if __name__ == '__main__':
    main()
