#!/usr/bin/env bash

isort --gitignore cf_genie/
autopep8 --exclude .env/ --recursive cf_genie/ --in-place
