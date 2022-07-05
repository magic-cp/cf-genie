#!/usr/bin/env bash

isort --gitignore task_runner.py cf_genie/
autopep8 --aggressive --aggressive --exclude .env/ --recursive task_runner.py cf_genie/ --in-place
