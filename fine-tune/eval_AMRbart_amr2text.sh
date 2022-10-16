#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

GPUID=$2
MODEL=$1
eval_beam=5
#modelcate=base
modelcate=large

lr=2e-6

datacate=AMR2.0
#datacate=AMR3.0
#datacate=AMR17-silver
#datacate=Giga
#datacate=New3
#datacate=TLP
#datacate=Bio

Tokenizer=../../../data/pretrained-model/bart-$modelcate          # replace this path with a directory containing standard BART tokenizer files,
# Tokenizer="facebook/bart-$modelcate"						                  # uncomment this line if your device has access to the Internet

OUTPUT_DIR_NAME=outputs/Eval-$datacate-AMRBart-$modelcate-amr2text-6taskPLM

OUTPUT_DIR=${ROOT_DIR}/${OUTPUT_DIR_NAME}
CACHE_DIR=${ROOT_DIR}/../data/$datacate/.cache

mkdir -p ${CACHE_DIR}
if [ ! -d ${OUTPUT_DIR} ];then
  mkdir -p ${OUTPUT_DIR}
else
  read -p "${OUTPUT_DIR} already exists, delete origin one [y/n]?" yn
  case $yn in
    [Yy]* ) rm -rf ${OUTPUT_DIR}; mkdir -p ${OUTPUT_DIR};;
    [Nn]* ) echo "exiting..."; exit;;
    * ) echo "Please answer yes or no.";;
  esac
fi

export OMP_NUM_THREADS=10
export CUDA_VISIBLE_DEVICES=${GPUID}
python -u ${ROOT_DIR}/run_amr2text.py \
    --data_dir ${ROOT_DIR}/../data/$datacate \
    --train_data_file ${ROOT_DIR}/../data/$datacate/train.jsonl \
    --eval_data_file ${ROOT_DIR}/../data/$datacate/val.jsonl \
    --test_data_file ${ROOT_DIR}/../data/$datacate/test.jsonl \
    --model_type ${MODEL} \
    --model_name_or_path ${MODEL} \
    --tokenizer_name_or_path ${Tokenizer} \
    --val_metric "bleu" \
    --learning_rate ${lr} \
    --max_epochs 1 \
    --max_steps -1 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --unified_input \
    --gpus 1 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --num_sanity_val_steps 0 \
    --src_block_size 1024 \
    --tgt_block_size 384 \
    --eval_max_length 384 \
    --eval_num_workers 4 \
    --process_num_workers 8 \
    --do_eval \
    --seed 42 \
    --fp16 \
    --eval_beam ${eval_beam} 2>&1 | tee $OUTPUT_DIR/eval.log
