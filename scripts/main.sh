#!/bin/bash

# Create summary plots of all datasets
./scripts/plot_datasets.sh

# Run hyperparameter search
# (Resulting hyperparameters have been set manually after and are currently hardcoded)
#./scripts/model_fit_hyperparams/adaboost.sh
#./scripts/model_fit_hyperparams/gradient_boosting.sh
#./scripts/model_fit_hyperparams/random_forest.sh

# Run all benchmark models
./scripts/model_fit_eval/mean.sh
./scripts/model_fit_eval/linear_trend.sh
./scripts/model_fit_eval/random_forest.sh
./scripts/model_fit_eval/gradient_boosting.sh
./scripts/model_fit_eval/adaboost.sh
./scripts/model_fit_eval/lstm.sh
./scripts/model_fit_eval/gru.sh
./scripts/model_fit_eval/cnn.sh

# Generate latex table rows containing MAE scores
./scripts/create_table.sh
