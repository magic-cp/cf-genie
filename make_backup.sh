#!/usr/bin/env bash

CURRENT_DATE=$(date -Iseconds)

BACKUP_DIR=backups/$CURRENT_DATE/
mkdir -p $BACKUP_DIR
time cp -r models temp dataset plots temp models hyper-parameters train-results grid-search-cv-results lstm-history $BACKUP_DIR
