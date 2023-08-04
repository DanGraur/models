#!/usr/bin/env bash

model=${1:-"yolo"}
dispatcher_address=${2:-""}

# Prepare the logging directory
log_dir=logs/${model}/$(date +%F_%H_%M_%S_%3N)
mkdir -p ${log_dir}

# Prepare the experiment parameters
additional_parameters=""
if [ "${model}" == "vit" ]; then
  experiment="deit_imagenet_pretrain"
  executable="official/projects/vit/train.py"

  # Run the experiment
  python3 ${executable} \
    --mode=train \
    --experiment=${experiment} \
    --model_dir=${log_dir} \
    --tpu="local" \
    --log_dir=${log_dir}/absl_out.log | tee ${log_dir}/console.log
fi


