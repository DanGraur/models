#!/usr/bin/env bash

model=${1:-"dlrm"}
batchsz=${2:-16384}

bucket="gs://sepehr-eu-logs/model_garden/bert"
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

elif [ "${model}" == "bert-local" ]; then
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

elif [ "${model}" == "bert-remote" ]; then
  experiment="bert/sentence_prediction"
  executable="official/nlp/train.py"
  task_config_file=`pwd`/"official/nlp/configs/experiments/glue_mnli_matched.yaml"
  model_config_file=`pwd`/"official/nlp/configs/models/bert_en_uncased_base.yaml"
  params_override="task.train_data.input_path=${bucket}/mnli_train.tf_record,
    task.validation_data.input_path=${bucket}/mnli_eval.tf_record,
    trainer.train_steps=2,
    trainer.validation_steps=4,
    ${base_params_override}
    "

elif [ "${model}" == "lambert" ]; then
  experiment="bert/sentence_prediction"
  executable="official/nlp/train.py"
  task_config_file=`pwd`/"official/nlp/configs/experiments/glue_mnli_matched.yaml"
  params_override="task.train_data.input_path=${bucket}/mnli_train.tf_record,
    task.validation_data.input_path=${bucket}/mnli_eval.tf_record,
    trainer.train_steps=2,
    trainer.validation_steps=4,
    task.hub_module_url=https://tfhub.dev/tensorflow/lambert_en_uncased_L-24_H-1024_A-16/1,
    ${base_params_override}
    "

elif [ "${model}" == "roberta" ]; then
  experiment="bert/sentence_prediction"
  executable="official/nlp/train.py"
  task_config_file=`pwd`/"official/nlp/configs/experiments/glue_mnli_matched.yaml"
  params_override="task.train_data.input_path=${bucket}/mnli_train.tf_record,
    task.validation_data.input_path=${bucket}/mnli_eval.tf_record,
    trainer.train_steps=2,
    trainer.validation_steps=4,
    task.hub_module_url=https://tfhub.dev/jeongukjae/roberta_en_cased_L-12_H-768_A-12/1,
    ${base_params_override}
    "

else
  echo "Model ${model} is not supported. Only valid options are: \"resnet\" and \"bert\"!"
fi

# Prepare the logging directory
log_dir=logs/${model}/$(date +%F_%H_%M_%S_%3N)
mkdir -p ${log_dir}

# Run the experiment
taskset -c 0-47 python3 ${executable} \
  --mode=train \
  --experiment=${experiment} \
  --config_file=${task_config_file} \
  --config_file=${model_config_file} \
  --model_dir=${log_dir} \
  --tpu=local \
  --params_override="${params_override}"