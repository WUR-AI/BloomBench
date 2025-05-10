#!/bin/bash

source scripts/config.sh

for seed in "${seeds[@]}"
do
  for dataset_key in "${dataset_keys[@]}"
  do
    run_prefix="fit_eval"
    dataset_time_step="daily"
    data_keys=""
    model_cls="MeanModel"

    model_name="${model_cls}_${dataset_key}_seed_${seed}"
    run_name="${run_prefix}_${model_name}"

    cmd=(
      "${python_key}" -m
      run.model_fit_eval
      --run_name "${run_name}"
      --dataset_name "${dataset_key}"
      --dataset_time_step "${dataset_time_step}"
      --data_keys ${data_keys}
      --skip_meteo_data_check
      --model_cls "${model_cls}"
      --model_name "${model_name}"
      --seed "${seed}"
      )

    # Run the command
    "${cmd[@]}"
  done
done
