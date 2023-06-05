#!/usr/bin/env bash

model=${1:-"resnet"}
dispatcher_address=${2:-""}

# Prepare the experiment parameters
additional_parameters=""
if [ "${model}" == "resnet" ]; then
  experiment="resnet_imagenet"
  executable="official/vision/train.py"
  if [ -z "${dispatcher_address}" ]; then
    config_file=`pwd`"/official/vision/configs/experiments/image_classification/imagenet_resnet50_tpu_without_service.yaml"
  else
    config_file=`pwd`"/official/vision/configs/experiments/image_classification/imagenet_resnet50_tpu_with_service.yaml"
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
python ${executable} \
  --mode=train \
  --experiment=${experiment} \
  --config_file=${config_file} \
  --model_dir=${log_dir} \
  ${additional_parameters}