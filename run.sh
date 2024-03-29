#!/usr/bin/env bash

model=${1:-"dlrm"}
batchsz=${2:-16384}

# 7495680 is a constant for 16 instances per batch
step_count=$(( 680000 / (${batchsz} / 16) ))
# Prepare the experiment parameters
base_params_override="
  task.train_data.global_batch_size=${batchsz},
  trainer.train_steps= ${step_count}
"
additional_parameters=""
if [ "${model}" == "dlrm" ]; then
  experiment="dlrm_criteo"
  executable="official/recommendation/ranking/train.py"
  task_config_file=`pwd`"/official/recommendation/ranking/configs/yaml/dlrm_criteo_tpu.yaml"
  model_config_file=`pwd`"/official/recommendation/ranking/configs/yaml/dlrm_criteo_tpu.yaml"
  params_override="${base_params_override}"

elif [ "${model}" == "bert" ]; then
  experiment="bert/sentence_prediction_text"
  executable="official/nlp/train.py"
  task_config_file=`pwd`/"official/nlp/configs/experiments/glue_mnli_text.yaml"
  model_config_file=`pwd`/"official/nlp/configs/models/bert_en_uncased_base.yaml"
  params_override="task.train_data.tfds_data_dir=`pwd`/../tfds,
    task.train_data.vocab_file=`pwd`/../uncased_L-12_H-768_A-12/vocab.txt,
    task.validation_data.tfds_data_dir=`pwd`/../tfds,
    task.validation_data.vocab_file=`pwd`/../uncased_L-12_H-768_A-12/vocab.txt,
    trainer.train_steps=2,
    trainer.validation_steps=4,
    ${base_params_override}
    "

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
  --config_file=${task_config_file} \
  --config_file=${model_config_file} \
  --model_dir=${log_dir} \
  --tpu=local \
  --params_override="${params_override}"