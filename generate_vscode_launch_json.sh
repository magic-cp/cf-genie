#!/usr/bin/env bash

while read task; do
    # echo "Generating launch.json for task: $task"
    module_name=$(basename $task .py)
    json='{"type": "python","request": "launch","name": "'$module_name'", "module": "cf_genie.tasks.'$module_name'", "justMyCode": true}'
    tasks="$tasks""$json"","
    # echo $json
    # echo $tasks
    TASKS=$tasks
done <<< "$(ls ./cf_genie/tasks/*.py)"

# Add a custom one to run the current file
TASKS='['$TASKS'{"type": "python","request": "launch","name": "Run Current File", "module": "cf_genie.tasks.${fileBasenameNoExtension}", "justMyCode": true}]'

jq -n --indent 4 --arg version "0.2.0" --argjson configurations "$TASKS" '$ARGS.named' > .vscode/launch.json
