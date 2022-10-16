# AMRBART
The refactored implementation for ACL2022 paper "Graph Pre-training for AMR Parsing and Generation". You may find our paper [here](https://arxiv.org/pdf/2203.07836.pdf) (Arxiv). The original implementation is avaliable [here](https://github.com/goodbai-nlp/AMRBART/tree/acl2022)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-pre-training-for-amr-parsing-and-1/amr-to-text-generation-on-ldc2017t10)](https://paperswithcode.com/sota/amr-to-text-generation-on-ldc2017t10?p=graph-pre-training-for-amr-parsing-and-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-pre-training-for-amr-parsing-and-1/amr-to-text-generation-on-ldc2020t02)](https://paperswithcode.com/sota/amr-to-text-generation-on-ldc2020t02?p=graph-pre-training-for-amr-parsing-and-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-pre-training-for-amr-parsing-and-1/amr-parsing-on-ldc2017t10)](https://paperswithcode.com/sota/amr-parsing-on-ldc2017t10?p=graph-pre-training-for-amr-parsing-and-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-pre-training-for-amr-parsing-and-1/amr-parsing-on-ldc2020t02)](https://paperswithcode.com/sota/amr-parsing-on-ldc2020t02?p=graph-pre-training-for-amr-parsing-and-1)

# Requirements
+ python 3.8
+ pytorch 1.8
+ transformers 4.21.3
+ datasets 2.4.0
+ Tesla V100 or A100

We recommend to use conda to manage virtual environments:
```
conda env update --name <env> --file requirements.yml
```

# Data Processing

<!-- Since AMR corpus require LDC license, we upload some examples for format reference. If you have the license, feel free to contact us for getting the preprocessed data. -->
You may download the AMR corpora at [LDC](https://www.ldc.upenn.edu).

Please follow [this respository](https://github.com/goodbai-nlp/AMR-Process) to preprocess AMR graphs:
``` 
bash run-preprocess-acl2022.sh
```


# Pre-training
```
bash run-posttrain-bart-textinf-joint-denoising-6task-large-unified-V100.sh /path/to/BART/
```

# Fine-tuning

For **AMR Parsing**, run
```
bash train-AMRBART-large-AMRParsing.sh /path/to/pre-trained/AMRBART/
```

For **AMR-to-text Generation**, run
```
bash train-AMRBART-large-AMR2Text.sh /path/to/pre-trained/AMRBART/
```


# Evaluation
```
cd evaluation
```

For **AMR Parsing**, run
```
bash eval_smatch.sh /path/to/gold-amr /path/to/predicted-amr
```
For better results, you can postprocess the predicted AMRs using the [BLINK](https://github.com/facebookresearch/BLINK) tool following [SPRING](https://github.com/SapienzaNLP/spring).

For **AMR-to-text Generation**, run
```
bash eval_gen.sh /path/to/gold-text /path/to/predicted-text
```

# Inference on your own data

If you want to run our code on your own data, try to transform your data into the format [here](https://github.com/muyeby/AMRBART/tree/main/examples), then run 

For **AMR Parsing**, run
```
bash inference_amr.sh /path/to/fine-tuned/AMRBART/
```

For **AMR-to-text Generation**, run
```
bash inference_text.sh /path/to/fine-tuned/AMRBART/
```

# Pre-trained Models

## Pre-trained AMRBART


|Setting| Params | checkpoint |
|  :----:  | :----:  |:---:|
| AMRBART-large | 409M | [model](todo) |


## Fine-tuned models on AMR-to-Text Generation

|Setting|  BLEU(JAMR_tok)  | Sacre-BLEU | checkpoint | output | 
|  :----:  | :----:  |:---:|  :----:  | :----:  |
| AMRBART-large (AMR2.0)  | 50.76 | 50.44 | [model](todo) | [output](todo) |
| AMRBART-large (AMR3.0) | 50.29 | 50.38 | [model](todo) | [output](todo) |

To get the tokenized bleu score, you need to use the scorer we provide [here](https://github.com/muyeby/AMRBART/blob/main/fine-tune/evaluation/eval_gen.sh). We use this script in order to ensure comparability with previous approaches.

## Fine-tuned models on AMR Parsing

|Setting|  Smatch | checkpoint | output | 
|  :----:  | :----:  |:---:|  :----:  |
| AMRBART-large (AMR2.0)  | 85.6 | [model](todo) | [output](todo) |
| AMRBART-large (AMR3.0)  | 84.3 | [model](todo) | [output](todo) |


# Acknowledgements
We thank authors of [SPRING](https://github.com/SapienzaNLP/spring), [amrlib](https://github.com/bjascob/amrlib), and [BLINK](https://github.com/facebookresearch/BLINK) that share open-source scripts for this project.
# References
```
@inproceedings{bai-etal-2022-graph,
    title = "Graph Pre-training for {AMR} Parsing and Generation",
    author = "Bai, Xuefeng  and
      Chen, Yulong  and
      Zhang, Yue",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.415",
    pages = "6001--6015"
}
```