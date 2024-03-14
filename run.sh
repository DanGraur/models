#!/usr/bin/env bash

model=${1:-"dlrm"}
batchsz=${2:-16384}

# 7495680 is a constant for 16 instances per batch
step_count=$(( 7495680 / (${batchsz} / 16) ))

# Prepare the experiment parameters
additional_parameters=""
if [ "${model}" == "dlrm" ]; then
  experiment="dlrm_criteo"
  executable="official/recommendation/ranking/train.py"
  config_file=`pwd`"/official/recommendation/ranking/configs/yaml/dlrm_criteo_tpu.yaml"
else
  echo "Model ${model} is not supported. Only valid options are: \"resnet\" and \"bert\"!"
fi

# Prepare the logging directory
log_dir=logs/${model}/$(date +%F_%H_%M_%S_%3N)
mkdir -p ${log_dir}

# Run the experiment
python3 ${executable} \
  --mode=train \
  --experiment=${experiment} \
  --config_file=${config_file} \
  --model_dir=${log_dir} \
  --tpu=local \
  ${additional_parameters} \
  --params_override="
task:
    train_data:
        global_batch_size: ${batchsz}
trainer:
  train_steps: ${step_count}
"