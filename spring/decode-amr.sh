#python bin/predict_sentences.py \
CUDA_VISIBLE_DEVICES=0 python bin/predict_amrs.py \
	--datasets data/LDC2017/amrs/split/test/*.txt \
	--gold-path data/tmp/gold.amr.txt \
	--pred-path data/tmp/pred.amr.txt \
	--checkpoint $1 \
	--beam-size 5 \
	--batch-size 5000 \
	--device "cuda" \
	--penman-linearization \
	--use-pointer-tokens
