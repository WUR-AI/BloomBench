##!/bin/bash
#
#seed=1
#
#run_prefix="fit_eval_hyperparameters"
#dataset_time_step="daily"
#data_keys="temperature_2m_mean daylight_duration"
#model_cls="RandomForestModel"
#
#model_name="${model_cls}_${dataset_key}_seed_${seed}"
#run_name="${run_prefix}_${model_name}"
#
#cmd=(
#  "${python_key}" -m
#  run.model_fit_eval
#  --run_name "${run_name}"
#  --dataset_name "${dataset_key}"
#  --dataset_time_step "${dataset_time_step}"
#  --data_keys ${data_keys}
#  --skip_meteo_data_check
#  --model_cls "${model_cls}"
#  --model_name "${model_name}"
#  --seed "${seed}"
##    --fit_hyperparameters
#  )
#
## Run the command
#"${cmd[@]}"
