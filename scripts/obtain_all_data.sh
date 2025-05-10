#!/bin/bash

source scripts/config.sh

${python_key} -m run.obtain_all_data --dataset_time_step daily
#${python_key} -m run.obtain_all_data --dataset_time_step hourly

./scripts/input_to_exit.sh

