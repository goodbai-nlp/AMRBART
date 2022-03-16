# AMRBART
An implementation for ACL2022 paper "Graph Pre-training for AMR Parsing and Generation". You may find our paper [here](https://arxiv.org/pdf/2203.07836.pdf) (Arxiv).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-pre-training-for-amr-parsing-and-1/amr-to-text-generation-on-ldc2017t10)](https://paperswithcode.com/sota/amr-to-text-generation-on-ldc2017t10?p=graph-pre-training-for-amr-parsing-and-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-pre-training-for-amr-parsing-and-1/amr-to-text-generation-on-ldc2020t02)](https://paperswithcode.com/sota/amr-to-text-generation-on-ldc2020t02?p=graph-pre-training-for-amr-parsing-and-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-pre-training-for-amr-parsing-and-1/amr-parsing-on-ldc2017t10)](https://paperswithcode.com/sota/amr-parsing-on-ldc2017t10?p=graph-pre-training-for-amr-parsing-and-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-pre-training-for-amr-parsing-and-1/amr-parsing-on-ldc2020t02)](https://paperswithcode.com/sota/amr-parsing-on-ldc2020t02?p=graph-pre-training-for-amr-parsing-and-1)

# Requirements
+ python 3.8
+ pytorch 1.8
+ transformers 4.8.2
+ pytorch-lightning 1.5.0
+ Tesla V100 or A100

We recommend to use conda to manage virtual environments:
```
conda env update --name <env> --file requirements.txt
```
We also provide a docker image [here](todo).

# Data 
<!-- Since AMR corpus require LDC license, we upload some examples for format reference. If you have the license, feel free to contact us for getting the preprocessed data. -->
You may download the AMR corpora at [LDC](https://www.ldc.upenn.edu).

# Pre-training
```
bash run-posttrain-bart-textinf-joint-denoising-6task-large-unified-V100.sh /path/to/BART/
```

# Fine-tuning

## AMR Parsing
```
bash finetune_AMRbart_amrparsing.sh /path/to/pre-trained/AMRBART/ gpu_id
```

## AMR-to-text Generation
```
bash finetune_AMRbart_amr2text.sh /path/to/pre-trained/AMRBART/ gpu_id
```


# Evaluation
## AMR Parsing
```
bash eval_AMRbart_amrparsing.sh /path/to/fine-tuned/AMRBART/ gpu_id
```

## AMR-to-text Generation
```
bash eval_AMRbart_amr2text.sh /path/to/fine-tuned/AMRBART/ gpu_id
```


# Pre-trained Models

## Pre-trained AMRBART


|Setting| Params | checkpoint |
|  :----:  | :----:  |:---:|
| AMRBART-base  | 142M | [model](https://1drv.ms/u/s!ArC7JSpdBblgpBZaL8-lr8wThJH7?e=7nW3Gm) |
| AMRBART-large | 409M | [model](https://1drv.ms/u/s!ArC7JSpdBblgpC2Fnu0JFCLYHC8f?e=jA1wte) |


## Fine-tuned models on AMR-to-Text Generation

|Setting|  BLEU(tok)  | BLEU(detok) | checkpoint | output | 
|  :----:  | :----:  |:---:|  :----:  | :----:  |
| AMRBART-large (AMR2.0)  | 49.8 | 45.7 | [model](https://1drv.ms/u/s!ArC7JSpdBblgpHmdWyLl0h33iHGH?e=c3eGc3) | [output](todo) |
| AMRBART-large (AMR3.0) | 49.2 | 45.0 | [model](https://1drv.ms/u/s!ArC7JSpdBblgpSfZv9epNcNzelij?e=0Pkj8V) | [output](todo) |

## Fine-tuned models on AMR Parsing

|Setting|  Smatch | checkpoint | output | 
|  :----:  | :----:  |:---:|  :----:  |
| AMRBART-large (AMR2.0)  | 85.4 | [model](https://1drv.ms/u/s!ArC7JSpdBblgpmeACrXnm6pS9SSN?e=hOQ5Xz) | [output](todo) |
| AMRBART-large (AMR3.0)  | 84.2 | [model](https://1drv.ms/u/s!ArC7JSpdBblgpw4c1qK3YEjVPr9P?e=DWfWf8) | [output](todo) |


# Todo
+ clean code


# References
```
@inproceedings{bai-etal-2022-graph,
    title = "Graph Pre-training for {AMR} Parsing and Generation",
    author = "Bai, Xuefeng  and
      Chen, Yulong and
      Zhang, Yue",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "todo",
    doi = "todo",
    pages = "todo"
}
```
