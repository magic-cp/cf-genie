#!/usr/bin/env bash

echo "Starting hyperopt-mongo-worker..."
while true
do
    PYTHONPATH=. hyperopt-mongo-worker --mongo="localhost:27017/cf_genie" --workdir=$(pwd) --max-consecutive-failures=10 --poll-interval=10
done;

