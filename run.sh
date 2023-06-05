#!/usr/bin/env bash

model=${1:-"resnet"}
dispatcher_address=${2:-""}

# Prepare the experiment parameters
additional_parameters=""
if [ "${model}" == "resnet" ]; then
  experiment="resnet_imagenet"
  executable="official/vision/train.py"
  base_path=`pwd`"/official/vision/configs/experiments/image_classification"
  config_file="${base_path}/imagenet_resnet50_tpu_opensource_base.yaml"
  if [ -z "${dispatcher_address}" ]; then
    override_file="${base_path}/imagenet_resnet50_tpu_without_service.yaml"
  elif [ "${dispatcher_address}" == "ideal" ]; then
    override_file="${base_path}/imagenet_resnet50_tpu_ideal_time.yaml"
  else
    override_file="${base_path}/magenet_resnet50_tpu_with_service.yaml"
    additional_parameters="--tf_data_service=\"${dispatcher_address}\""
  fi
elif [ "${model}" == "bert" ]; then
  echo "BERT is not supported yet!"
else
  echo "Model ${model} is not supported. Only valid options are: \"resnet\" and \"bert\"!"
fi

# Prepare the logging directory
deployment_type=$( if [ -z "${dispatcher_address}" ]; then echo "collocated"; else echo "disaggregated"; fi )
log_dir=logs/${model}/${deployment_type}_$(date +%F_%H_%M_%S_%3N)
mkdir -p ${log_dir}

# Run the experiment
python3 --log_dir=${log_dir}/absl_out.log \
  ${executable} \
  --mode=train \
  --experiment=${experiment} \
  --config_file=${config_file} \
  --params_override=${override_file} \
  --model_dir=${log_dir} \
  --tpu="local" \
  ${additional_parameters} | tee ${log_dir}/console.log