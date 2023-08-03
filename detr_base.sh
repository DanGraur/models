#!/bin/bash

log_dir=logs/detr/collocated_$(date +%F_%H_%M_%S_%3N)
mkdir -p ${log_dir}

python3 official/projects/detr/train.py \
  --experiment=detr_coco_tfrecord \
  --mode=train \
  --model_dir=${log_dir}
#  --params_override=task.init_checkpoint='gs://tf_model_garden/vision/resnet50_imagenet/ckpt-62400',trainer.train_steps=554400,trainer.optimizer_config.learning_rate.stepwise.boundaries="[369600]"
