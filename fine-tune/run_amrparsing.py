# Copyright 2021 Xuefeng BAI
# Contact: xfbai.hk@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import glob
import argparse
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from model_interface.model_amrparsing import AMRParsingModelModule
from data_interface.dataset_pl import AMRParsingDataModule
from common.options import add_model_specific_args
from common.callbacks import LoggingCallback, get_early_stopping_callback, get_checkpoint_callback
from transformers import AutoConfig
from spring_amr.tokenization_bart import PENMANBartTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_callbacks(args, model):
    callbacks = []
    callbacks.append(LoggingCallback())

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
        callbacks.append(es_callback)

    lower_is_better = args.val_metric == "loss"
    checkpoint_callback = get_checkpoint_callback(
        args.output_dir,
        model.val_metric,
        save_top_k=args.save_total_limit,
        lower_is_better=lower_is_better,
        save_interval=args.save_interval,
    )
    callbacks.append(checkpoint_callback)

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(logging_interval="step"))

    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    odir = Path(args.output_dir)
    odir.mkdir(exist_ok=True)
    if args.resume:
        checkpoints = list(
            sorted(glob.glob(os.path.join(args.output_dir, "last.ckpt"), recursive=True))
        )
        assert (
            len(checkpoints) >= 1
        ), f"No checkpoints founded at {os.path.join(args.output_dir, 'last.ckpt')}"
        load_path = checkpoints[-1]
    else:
        load_path = None

    tokenizer_name_or_path = args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path
    amr_tokenizer = PENMANBartTokenizer.from_pretrained(
        tokenizer_name_or_path,
        collapse_name_ops=False,
        use_pointer_tokens=True,
        raw_graph=False,
    )
    data_module = AMRParsingDataModule(amr_tokenizer, **vars(args))
    data_module.setup()
    args.train_dataset_size = len(data_module.train_dataset)

    if load_path is None:
        model = AMRParsingModelModule(amr_tokenizer, args)
    else:                                   # to test
        model = AMRParsingModelModule(amr_tokenizer, args)
        args.resume_from_checkpoint = load_path

    print(model.model)
    print(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.model.parameters()),
            sum(p.numel() for p in model.model.parameters() if p.requires_grad),
        )
    )

    # # If you want to change the logger's saving folder
    logger = TensorBoardLogger(save_dir="exp_log", name=args.output_dir)
    args.logger = logger
    args.callbacks = load_callbacks(args, model)

    train_params = {}
    if args.fp16:
        train_params["precision"] = 16
        # train_params["amp_backend"] = "apex"
        # train_params["amp_level"] = args.fp16_opt_level
    if args.gpus > 1:
        train_params["accelerator"] = "gpu"
        train_params["strategy"] = "ddp"

    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches
    # train_params["deterministic"] = True
    trainer = Trainer.from_argparse_args(args, **train_params)

    if args.do_train:
        print("Start Training ...")
        # model.setup(stage='fit')
        trainer.fit(model, datamodule=data_module)

    if not (args.do_predict or args.do_eval):
        return model
    
    if not args.do_train:
        if args.do_predict:
            print("Evaluation on valid Set ...")
            trainer.validate(model, datamodule=data_module)
            print("Evaluation on test Set ...")
            trainer.test(model, datamodule=data_module)
        if args.do_eval:
            print("Evaluation on test Set ...")
            trainer.test(model, datamodule=data_module)
        return model

    trainer.test(model, datamodule=data_module)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_model_specific_args(parser, os.getcwd())
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
