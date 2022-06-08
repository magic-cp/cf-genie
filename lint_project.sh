#!/usr/bin/env bash

isort --gitignore cf_genie/
autopep8 --aggressive --aggressive --exclude .env/ --recursive cf_genie/ --in-place
