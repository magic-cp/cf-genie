#!/usr/bin/env bash

echo "Starting hyperopt-mongo-worker..."
PYTHONPATH=. hyperopt-mongo-worker --mongo="localhost:27017/cf_genie" --workdir=$(pwd)
