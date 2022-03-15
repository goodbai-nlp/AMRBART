#!/bin/bash

python bin/predict_amrs.py \
    --datasets /nfs/Desktop/data/LittlePrince.txt \
    --gold-path data/tmp/LittlePrince/gold.amr.txt \
    --pred-path data/tmp/LittlePrince/pred.amr.txt \
    --checkpoint runs/0/best-smatch_checkpoint_18_0.8421.pt \
    --beam-size 5 \
    --batch-size 2000 \
    --device cuda \
    --penman-linearization --use-pointer-tokens