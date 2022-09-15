#!/usr/bin/env bash

DATASET_GENERATION_TASKS="
load_cf_data
scrap_dataset
cleanup_dataset
"
function run_tasks() {
    for task in $1; do
        echo "Running $task"
    done
}


run_tasks "$DATASET_GENERATION_TASKS"
