# !/bin/bash
CUDA_VISIBLE_DEVICES=0 python -u bin/train.py --config configs/config-large.yaml --direction amr 2>&1 | tee run-amrparsing-bartlarge.log 
