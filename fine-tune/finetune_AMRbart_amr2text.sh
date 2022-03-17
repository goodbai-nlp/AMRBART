#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

GPUID=$2
MODEL=$1
eval_beam=5
modelcate=base

lr=5e-6

datacate=AMR2.0
#datacate=AMR3.0

Tokenizer=../../../data/pretrained-model/bart-$modelcate

export OUTPUT_DIR_NAME=outputs/$datacate-AMRBart-$modelcate-amr2text-6taskPLM-5e-5-finetune-lr${lr}

export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
cache=../../../data/.cache/

if [ ! -d $OUTPUT_DIR ];then
  mkdir -p $OUTPUT_DIR
else
  echo "${OUTPUT_DIR} already exists, change a new one or delete origin one"
  exit 0
fi

export OMP_NUM_THREADS=10
export CUDA_VISIBLE_DEVICES=${GPUID}
python -u ${ROOT_DIR}/run_amr2text.py \
    --data_dir=../data/$datacate \
    --train_data_file=../data/$datacate/train.jsonl \
    --eval_data_file=../data/$datacate/val.jsonl \
    --test_data_file=../data/$datacate/test.jsonl \
    --model_type ${MODEL} \
    --model_name_or_path=${MODEL} \
    --tokenizer_name_or_path=${Tokenizer} \
    --val_metric "bleu" \
    --learning_rate=${lr} \
    --max_epochs 20 \
    --max_steps -1 \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=4 \
    --unified_input \
    --accumulate_grad_batches 1 \
    --early_stopping_patience 10 \
    --gpus 1 \
    --output_dir=${OUTPUT_DIR} \
    --cache_dir ${cache} \
    --num_sanity_val_steps 4 \
    --src_block_size=1024 \
    --tgt_block_size=384 \
    --eval_max_length=384 \
    --train_num_workers 16 \
    --eval_num_workers 4 \
    --process_num_workers 8 \
    --do_train --do_predict \
    --seed 42 \
    --fp16 \
    --eval_beam ${eval_beam} 2>&1 | tee $OUTPUT_DIR/run.log
