
#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

GPUID=$2
MODEL=$1
eval_beam=5
modelcate=base
# modelcate=large

lr=8e-6

datacate=DataToParse                                              # replace this path with your own data. 

Tokenizer=../../../data/pretrained-model/bart-$modelcate          # replace this path with a directory containing standard BART tokenizer files

export OUTPUT_DIR_NAME=outputs/Eval-${datacate}-AMRBart-${modelcate}-amrparsing-6taskPLM

export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
cache=../../../data/.cache/

if [ ! -d $OUTPUT_DIR ];then
  mkdir -p $OUTPUT_DIR
else
  read -p "${OUTPUT_DIR} already exists, delete origin one [y/n]?" yn
  case $yn in
    [Yy]* ) rm -rf $OUTPUT_DIR; mkdir -p $OUTPUT_DIR;;
    [Nn]* ) echo "exiting..."; exit;;
    * ) echo "Please answer yes or no.";;
  esac
fi

export OMP_NUM_THREADS=10
export CUDA_VISIBLE_DEVICES=${GPUID}
python -u ${ROOT_DIR}/inference_amr.py \
    --data_dir=../data/$datacate \
    --train_data_file=../data/$datacate/train.jsonl \
    --eval_data_file=../data/$datacate/val.jsonl \
    --test_data_file=../data/$datacate/test.jsonl \
    --model_type ${MODEL} \
    --model_name_or_path=${MODEL} \
    --tokenizer_name_or_path=${Tokenizer} \
    --val_metric "smatch" \
    --learning_rate=${lr} \
    --max_epochs 1 \
    --max_steps -1 \
    --per_gpu_eval_batch_size=48 \
    --unified_input \
    --gpus 1 \
    --output_dir=${OUTPUT_DIR} \
    --cache_dir ${cache} \
    --num_sanity_val_steps 0 \
    --src_block_size=256 \
    --tgt_block_size=512 \
    --eval_max_length=512 \
    --eval_num_workers 2 \
    --process_num_workers 8 \
    --do_eval \
    --seed 42 \
    --fp16 \
    --eval_beam ${eval_beam} 2>&1 | tee $OUTPUT_DIR/Eval.log
