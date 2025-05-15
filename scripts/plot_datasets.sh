#!/bin/bash

source scripts/config.sh

${python_key} -m run.plot_datasets
${python_key} -m run.plot_dataset_cv