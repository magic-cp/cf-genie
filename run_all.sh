#!/usr/bin/env bash

DOWNLOAD_CF_TASKS="
load_cf_data
generate_temp_input_for_raw_dataset
scrap_dataset"

DATASET_GENERATION_TASKS="
cleanup_dataset
split_training_and_test
balance_dataset
embed_datasets
"

PLOTTING_AND_ANALYSIS_TASKS="
generate_wordclouds
plot_statistics_for_tag_group
result_analysis
run_stats_on_embedded_data
"

function run_tasks() {
    for task in $1; do
        echo "Running $task"
    done
}


run_tasks "$DATASET_GENERATION_TASKS"
