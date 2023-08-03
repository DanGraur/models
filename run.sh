#!/usr/bin/env bash

model=${1:-"yolo"}
dispatcher_address=${2:-""}

# Prepare the experiment parameters
additional_parameters=""
if [ "${model}" == "yolo" ]; then
  experiment="resnet_imagenet"
  executable="official/projects/yolo/train.py"
  base_path=`pwd`"/official/projects/yolo/configs/experiments/yolov4/detection"
  config_file="${base_path}/yolov4_512_tpu_base.yaml"
  if [ -z "${dispatcher_address}" ]; then
    deployment_type="collocated"
    override_file="${base_path}/imagenet_resnet50_tpu_without_service.yaml"
  else
    echo "Dispatcher address provided but no support for it currently!"
    exit 1
#    deployment_type="disaggregated"
#    override_file="${base_path}/imagenet_resnet50_tpu_with_service.yaml"
#    additional_parameters="--tf_data_service=grpc://${dispatcher_address}:31000"
  fi
fi
#elif [ "${model}" == "bert" ]; then
#  experiment="bert/pretraining"
#  executable="official/nlp/train.py"
#  base_path=`pwd`"/official/nlp/configs/experiments"
#  config_file="${base_path}/bert_pretraining_opensource_base.yaml"
#  if [ -z "${dispatcher_address}" ]; then
#    deployment_type="non_coordinated"
#    override_file="${base_path}/bert_pretraining_non_coordinated.yaml"
#  else
#    deployment_type="coordinated"
#    override_file="${base_path}/bert_pretraining_coordinated.yaml"
#    additional_parameters="--tf_data_service=grpc://${dispatcher_address}:31000"
#  fi
#elif [ "${model}" == "glue" ]; then
#  experiment="bert/sentence_prediction_text"
#  executable="official/nlp/train.py"
#  base_path=`pwd`"/official/nlp/configs/experiments"
#  config_file="${base_path}/glue_opensource_base.yaml"
#  if [ -z "${dispatcher_address}" ]; then
#    deployment_type="non_coordinated"
#    override_file="${base_path}/bert_pretraining_non_coordinated.yaml"
#  else
#    deployment_type="coordinated"
#    override_file="${base_path}/bert_pretraining_coordinated.yaml"
#    additional_parameters="--tf_data_service=grpc://${dispatcher_address}:31000"
#  fi
#else
#  echo "Model ${model} is not supported. Only valid options are: \"resnet\", \"glue\", and \"bert\"!"
#fi

# Prepare the logging directory
log_dir=logs/${model}/${deployment_type}_$(date +%F_%H_%M_%S_%3N)
mkdir -p ${log_dir}

# Run the experiment
python3 ${executable} \
  --mode=train \
#  --experiment=${experiment} \
  --config_file=${config_file} \
#  --params_override=${override_file} \
  --model_dir=${log_dir} \
  --tpu="local" \
  --log_dir=${log_dir}/absl_out.log \
#  ${additional_parameters} | tee ${log_dir}/console.log
  | tee ${log_dir}/console.log