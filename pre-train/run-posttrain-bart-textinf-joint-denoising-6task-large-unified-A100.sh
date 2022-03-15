export CUDA_VISIBLE_DEVICES=0,1
dataset=Giga
datapath=../data/$dataset
MODEL=$1
interval=1

lr=5e-5

model_size="large"

outpath=output/${dataset}-bart-${model_size}-Unifiedtextinf-JointDenoise-6task-${lr}-AMREOS

mkdir -p $outpath
echo "OutputDir: $outpath"

python -u -m torch.distributed.launch --nproc_per_node=2 run_multitask_unified_pretraining.py \
  --train_file $datapath/train.jsonl \
  --val_file $datapath/val.jsonl \
  --test_file $datapath/test.jsonl \
  --output_dir $outpath \
  --mlm \
  --mlm_amr \
  --mlm_text \
  --mlm_amr_plus_text \
  --mlm_text_plus_amr \
  --mlm_joint_to_amr \
  --mlm_joint_to_text \
  --block_size 512 \
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --model_type "facebook/bart-${model_size}" \
  --model_name_or_path $MODEL \
  --save_total_limit 2 \
  --do_train \
  --do_eval \
  --evaluate_during_training  \
  --num_train_epochs 100  \
  --learning_rate $lr \
  --joint_train_interval $interval \
  --warmup_steps 2500 \
  --max_steps 100000 \
  --logging_steps 1000 \
  --fp16 \
  --overwrite_output_dir 2>&1 | tee $outpath/run.log
