#!/bin/bash

source scripts/config.sh

for seed in "${seeds[@]}"
do
  for dataset_key in "${dataset_keys[@]}"
    do
      run_prefix="fit_eval"
      dataset_time_step="daily"
      data_keys="temperature_2m_mean daylight_duration"
      model_cls="CNNModel"

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
        --batch_size 64
        --num_epochs 1000
        --optimizer adam
        --lr 1e-3
        --weight_decay 1e-4
        --scheduler_step_size 100
        --scheduler_decay 0.9
        --early_stopping
        --early_stopping_patience 5
        --early_stopping_min_delta 1e-3
        --validation_period 10
        )

      # Run the command
      "${cmd[@]}"
  done
done
