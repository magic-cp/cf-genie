#!/usr/bin/env bash

echo "Checking imports..."
isort --gitignore -c cf_genie/

echo "Checking code style..."
pycodestyle cf_genie/
