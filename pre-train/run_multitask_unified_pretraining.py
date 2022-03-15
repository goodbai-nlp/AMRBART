### THIS FILE IS COPIED FROM THE HUGGINGFACE REPOSITORY FOR CONVENIENCE.

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import argparse
import glob
import json
import logging
import os
import math
import pickle
from posixpath import join
import random
import re
import shutil
from typing import Dict, List, Any, Tuple

from numpy.lib.twodim_base import mask_indices

from dataset import AMRDataSetFast, DataCollatorForSeq2Seq

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from spring_amr.tokenization_bart import PENMANBartTokenizer
from model_utils import (
    freeze_params,
    freeze_embeds,
    assert_all_frozen,
    get_inverse_sqrt_schedule_with_warmup,
    activate_embeds,
)

from model_utils import (
    get_STD2partial,
    get_MTEG2text,
    get_ETMG2graph,
    get_PTPG2partial,
    get_MTMG2partial,
    get_MTMG2TG,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

# from transformers.modeling_auto import MODEL_WITH_LM_HEAD_MAPPING

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

# MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            "Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint)
        )
        shutil.rmtree(checkpoint)


def train(
    args,
    train_dataset,
    eval_dataset,
    collate_fn,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config,
) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = (
        RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=6,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    model = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    # model.resize_token_embeddings(len(tokenizer))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # )
    scheduler = get_inverse_sqrt_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    epoch_step = 0
    steps_trained_in_current_epoch = 0
    best_score = float("inf")
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
            )

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info(
                "  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch
            )
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss, epoch_loss = 0.0, 0.0, 0.0
    tr_fisher_loss, logging_fisher_loss = 0.0, 0.0

    model.zero_grad()

    # fisher, opt_param = get_fisher(train_dataloader, args, tokenizer, model, config)

    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproducibility
    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )

        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            if args.mlm_amr:
                # masked_input, attention_mask, dec_input, labels = get_mlm_inputs(batch, tokenizer, args, inp='amr')
                # masked_input, attention_mask, dec_input, labels = get_text_infilling_inputs(batch, tokenizer, args, inp='amr')
                masked_input, attention_mask, dec_input, labels = get_ETMG2graph(
                    batch, tokenizer, mlm_prob=0.35
                )

                masked_input = masked_input.to(args.device)
                labels = labels.to(args.device)
                dec_input = dec_input.to(args.device)

                if step == 0 and epoch == 0:
                    save_dummy_batch2(
                        args, masked_input, dec_input, labels, tokenizer, prefix="Etextamr2amr"
                    )
                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input,
                    labels=labels,
                )
                amr_loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            else:
                amr_loss = 0

            if args.mlm_text:
                # masked_input, attention_mask, dec_input, labels = get_mlm_inputs(batch, tokenizer, args, inp='text')
                # masked_input, attention_mask, dec_input, labels = get_text_infilling_inputs(batch, tokenizer, args, inp='text')
                masked_input, attention_mask, dec_input, labels = get_MTEG2text(
                    batch, tokenizer, mlm_prob=0.35
                )

                masked_input = masked_input.to(args.device)
                labels = labels.to(args.device)
                dec_input = dec_input.to(args.device)
                if step == 0 and epoch == 0:
                    save_dummy_batch2(
                        args, masked_input, dec_input, labels, tokenizer, prefix="textEamr2text"
                    )
                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input,
                    labels=labels,
                )
                text_loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            else:
                text_loss = 0

            if args.mlm_text_plus_amr:
                if step % args.joint_train_interval == 0:
                    mlm_prob = 0.1 + global_step / args.max_steps * 0.75
                    masked_input, attention_mask, dec_input, labels = get_PTPG2partial(
                        batch, tokenizer, inp="text", mlm_prob=mlm_prob
                    )
                    masked_input = masked_input.to(args.device)
                    labels = labels.to(args.device)
                    dec_input = dec_input.to(args.device)
                    if step == 0 and epoch == 0:
                        save_dummy_batch2(
                            args,
                            masked_input,
                            dec_input,
                            labels,
                            tokenizer,
                            prefix="val_MtextAmr2text",
                        )
                    outputs = model(
                        input_ids=masked_input,
                        attention_mask=attention_mask,
                        decoder_input_ids=dec_input,
                        labels=labels,
                    )
                    text_joint_loss = outputs[0]
                else:
                    text_joint_loss = 0
            else:
                text_joint_loss = 0

            if args.mlm_amr_plus_text:
                if step % args.joint_train_interval == 0:
                    mlm_prob = 0.1 + global_step / args.max_steps * 0.75
                    masked_input, attention_mask, dec_input, labels = get_PTPG2partial(
                        batch, tokenizer, inp="amr", mlm_prob=mlm_prob
                    )
                    masked_input = masked_input.to(args.device)
                    labels = labels.to(args.device)
                    dec_input = dec_input.to(args.device)
                    if step == 0 and epoch == 0:
                        save_dummy_batch2(
                            args,
                            masked_input,
                            dec_input,
                            labels,
                            tokenizer,
                            prefix="val_TextMamr2amr",
                        )

                    outputs = model(
                        input_ids=masked_input,
                        attention_mask=attention_mask,
                        decoder_input_ids=dec_input,
                        labels=labels,
                    )
                    amr_joint_loss = outputs[0]
                else:
                    amr_joint_loss = 0
            else:
                amr_joint_loss = 0

            if args.mlm_joint_to_text:
                mlm_prob = 0.35
                masked_input, attention_mask, dec_input, labels = get_MTMG2partial(
                    batch, tokenizer, inp="text", mlm_prob=mlm_prob
                )
                masked_input = masked_input.to(args.device)
                labels = labels.to(args.device)
                dec_input = dec_input.to(args.device)
                if step == 0 and epoch == 0:
                    save_dummy_batch2(
                        args,
                        masked_input,
                        dec_input,
                        labels,
                        tokenizer,
                        prefix="val_MtextMamr2text",
                    )
                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input,
                    labels=labels,
                )
                text_joint_loss2 = outputs[0]
            else:
                text_joint_loss2 = 0

            if args.mlm_joint_to_amr:
                mlm_prob = 0.35
                masked_input, attention_mask, dec_input, labels = get_MTMG2partial(
                    batch, tokenizer, inp="amr", mlm_prob=mlm_prob
                )
                masked_input = masked_input.to(args.device)
                labels = labels.to(args.device)
                dec_input = dec_input.to(args.device)
                if step == 0 and epoch == 0:
                    save_dummy_batch2(
                        args, masked_input, dec_input, labels, tokenizer, prefix="val_MtextMamr2amr"
                    )

                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input,
                    labels=labels,
                )
                amr_joint_loss2 = outputs[0]
            else:
                amr_joint_loss2 = 0

            if args.mlm_joint_to_joint:
                mlm_prob = 0.35
                masked_input, attention_mask, dec_input, labels = get_MTMG2TG(
                    batch, tokenizer, mlm_prob=mlm_prob
                )
                masked_input = masked_input.to(args.device)
                labels = labels.to(args.device)
                dec_input = dec_input.to(args.device)
                if step == 0 and epoch == 0:
                    save_dummy_batch2(
                        args,
                        masked_input,
                        dec_input,
                        labels,
                        tokenizer,
                        prefix="val_MtextMamr2textamr",
                    )

                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input,
                    labels=labels,
                )
                joint2joint_loss = outputs[0]
            else:
                joint2joint_loss = 0

            loss = (
                amr_loss
                + text_loss
                + text_joint_loss
                + amr_joint_loss
                + text_joint_loss2
                + amr_joint_loss2
                + joint2joint_loss
            )

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # fisher_loss = torch.tensor(0.0, device=masked_input.device)
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         fisher_loss_p = (
            #             1000 * fisher[name].cuda() * (param - opt_param[name].cuda()).pow(2)
            #         )
            #         fisher_loss += fisher_loss_p.sum()
            # epoch_iterator.set_postfix(lm_loss=loss.item(), fisher_loss=fisher_loss.item(), lr=scheduler.get_lr()[0])
            # loss += fisher_loss
            epoch_iterator.set_postfix(lm_loss=loss.item(), lr=scheduler.get_lr()[0])

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            epoch_step += 1
            tr_loss += loss.item()
            epoch_loss += loss.item()
            # tr_fisher_loss += fisher_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if (
                        # args.local_rank == -1 and args.evaluate_during_training
                        args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(
                            args, eval_dataset, collate_fn, model, tokenizer, config=config
                        )
                        cur_score = results["perplexity"].item()
                        if cur_score < best_score:
                            best_score = cur_score
                            checkpoint_prefix = "checkpoint"
                            # Save model checkpoint
                            output_dir = os.path.join(
                                args.output_dir,
                                "{}-{}-{:.3f}".format(checkpoint_prefix, global_step, best_score),
                            )
                            os.makedirs(output_dir, exist_ok=True)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            _rotate_checkpoints(args, checkpoint_prefix)

                            torch.save(
                                optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                            )
                            torch.save(
                                scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                            )
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss", (tr_loss - logging_loss) / args.logging_steps, global_step
                    )
                    # tb_writer.add_scalar(
                    #     "fisher_loss",
                    #     (tr_fisher_loss - logging_fisher_loss) / args.logging_steps,
                    #     global_step,
                    # )
                    logging_loss = tr_loss
                    # logging_fisher_loss = tr_fisher_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            results = evaluate(args, eval_dataset, collate_fn, model, tokenizer, config=config)
            cur_score = results["perplexity"].item()
            checkpoint_prefix = "checkpoint"
            # Save model checkpoint
            output_dir = os.path.join(
                args.output_dir, "{}-last-{:.3f}".format(checkpoint_prefix, cur_score),
            )
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)
            train_iterator.close()
            break
        
        checkpoint_prefix = "checkpoint"
        output_dir = os.path.join(
            args.output_dir, "{}-last-epoch".format(checkpoint_prefix),
        )
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)
        avg_epoch_loss = epoch_loss / epoch_step
        logger.info("\nEpoch End... \navg_train_loss = %s", str(avg_epoch_loss))

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(
    args,
    eval_dataset,
    collate_fn,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config=None,
    prefix="",
) -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    pbar = tqdm(eval_dataloader, desc="Evaluating")
    for batch in pbar:

        with torch.no_grad():
            if args.mlm_amr:
                # masked_input, attention_mask, dec_input, labels = get_mlm_inputs(batch, tokenizer, args, inp='amr')
                # masked_input, attention_mask, dec_input, labels = get_text_infilling_inputs(batch, tokenizer, args, inp='amr')
                masked_input, attention_mask, dec_input, labels = get_ETMG2graph(
                    batch, tokenizer, mlm_prob=0.35
                )

                masked_input = masked_input.to("cuda:0")
                labels = labels.to("cuda:0")
                dec_input = dec_input.to("cuda:0")
                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input,
                    labels=labels,
                )
                amr_loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            else:
                amr_loss = 0

            if args.mlm_text:
                # masked_input, attention_mask, dec_input, labels = get_mlm_inputs(batch, tokenizer, args, inp='text')
                # masked_input, attention_mask, dec_input, labels = get_text_infilling_inputs(batch, tokenizer, args, inp='text')
                masked_input, attention_mask, dec_input, labels = get_MTEG2text(
                    batch, tokenizer, mlm_prob=0.35
                )

                masked_input = masked_input.to("cuda:0")
                labels = labels.to("cuda:0")
                dec_input = dec_input.to("cuda:0")
                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input,
                    labels=labels,
                )
                text_loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            else:
                text_loss = 0

            if args.mlm_text_plus_amr:
                # masked_input, attention_mask, dec_input, labels = get_mlm_joint_inputs_full(batch, tokenizer, args, inp='text')
                masked_input, attention_mask, dec_input, labels = get_PTPG2partial(
                    batch, tokenizer, inp="text"
                )
                masked_input = masked_input.to("cuda:0")
                labels = labels.to("cuda:0")
                dec_input = dec_input.to("cuda:0")
                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input,
                    labels=labels,
                )
                text_joint_loss = outputs[0]
            else:
                text_joint_loss = 0

            if args.mlm_amr_plus_text:
                # masked_input, attention_mask, dec_input, labels = get_mlm_joint_inputs_full(batch, tokenizer, args, inp='amr')
                masked_input, attention_mask, dec_input, labels = get_PTPG2partial(
                    batch, tokenizer, inp="amr"
                )
                masked_input = masked_input.to("cuda:0")
                labels = labels.to("cuda:0")
                dec_input = dec_input.to("cuda:0")
                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input,
                    labels=labels,
                )
                amr_joint_loss = outputs[0]
            else:
                amr_joint_loss = 0

            if args.mlm_joint_to_text:
                mlm_prob = 0.35
                masked_input, attention_mask, dec_input, labels = get_MTMG2partial(
                    batch, tokenizer, inp="text", mlm_prob=mlm_prob
                )
                masked_input = masked_input.to("cuda:0")
                labels = labels.to("cuda:0")
                dec_input = dec_input.to("cuda:0")
                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input,
                    labels=labels,
                )
                text_joint_loss2 = outputs[0]
            else:
                text_joint_loss2 = 0

            if args.mlm_joint_to_amr:
                mlm_prob = 0.35
                masked_input, attention_mask, dec_input, labels = get_MTMG2partial(
                    batch, tokenizer, inp="amr", mlm_prob=mlm_prob
                )
                masked_input = masked_input.to("cuda:0")
                labels = labels.to("cuda:0")
                dec_input = dec_input.to("cuda:0")
                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input,
                    labels=labels,
                )
                amr_joint_loss2 = outputs[0]
            else:
                amr_joint_loss2 = 0

            if args.mlm_joint_to_joint:
                mlm_prob = 0.35
                masked_input, attention_mask, dec_input, labels = get_MTMG2TG(
                    batch, tokenizer, mlm_prob=mlm_prob
                )
                masked_input = masked_input.to("cuda:0")
                labels = labels.to("cuda:0")
                dec_input = dec_input.to("cuda:0")
                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input,
                    labels=labels,
                )
                joint2joint_loss = outputs[0]
            else:
                joint2joint_loss = 0

            loss = (
                amr_loss
                + text_loss
                + text_joint_loss
                + amr_joint_loss
                + text_joint_loss2
                + amr_joint_loss2
                + joint2joint_loss
            )

            pbar.set_postfix(lm_loss=loss.mean().item())

            eval_loss += loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity, "eval_loss": eval_loss}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("\n***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def ids_to_clean_text(tokenizer, generated_ids: List[int]):
    generated_ids.masked_fill_(generated_ids == -100, tokenizer.pad_token_id)
    gen_text = tokenizer.batch_decode(generated_ids, clean_up_tokenization_spaces=False)
    return " ".join(gen_text)


def save_dummy_batch(args, batch, tokenizer):
    dummy_ids, dummy_tokens = [], []
    for idx in range(len(batch["input_ids"])):
        ith_dict, ith_tok_dict = {}, {}
        ith_dict["input_ids"] = batch["input_ids"][idx].tolist()
        ith_dict["label_ids"] = batch["labels"][idx].tolist()
        ith_dict["dec_inp_ids"] = batch["decoder_input_ids"][idx].tolist()
        # print("inp_ids", batch["input_ids"][idx])
        # print("label_ids", batch["labels"][idx])
        # print("dec_inp", batch["decoder_input_ids"][idx])
        dummy_ids.append(ith_dict)

        ith_tok_dict["input_tokens"] = ids_to_clean_text(tokenizer, batch["input_ids"][idx])
        ith_tok_dict["label_tokens"] = ids_to_clean_text(tokenizer, batch["labels"][idx])
        ith_tok_dict["dec_inp_tokens"] = ids_to_clean_text(
            tokenizer, batch["decoder_input_ids"][idx]
        )
        dummy_tokens.append(ith_tok_dict)

    with open(args.output_dir + "/dummy_ids.json", "w", encoding="utf-8") as fout:
        json.dump(dummy_ids, fout, indent=4)
    with open(args.output_dir + "/dummy_token.json", "w", encoding="utf-8") as fout:
        json.dump(dummy_tokens, fout, indent=4)


def save_dummy_batch2(args, input_ids, dec_inp_ids, labels, tokenizer, prefix="train"):
    dummy_ids, dummy_tokens = [], []
    for idx in range(len(input_ids)):
        ith_dict, ith_tok_dict = {}, {}
        ith_dict["input_ids"] = input_ids[idx].tolist()
        ith_dict["label_ids"] = labels[idx].tolist()
        ith_dict["dec_inp_ids"] = dec_inp_ids[idx].tolist()
        dummy_ids.append(ith_dict)

        ith_tok_dict["input_tokens"] = ids_to_clean_text(tokenizer, input_ids[idx])
        ith_tok_dict["label_tokens"] = ids_to_clean_text(tokenizer, labels[idx])
        ith_tok_dict["dec_inp_tokens"] = ids_to_clean_text(tokenizer, dec_inp_ids[idx])
        dummy_tokens.append(ith_tok_dict)

    with open(args.output_dir + f"/dummy_{prefix}_ids.json", "w", encoding="utf-8") as fout:
        json.dump(dummy_ids, fout, indent=4)
    with open(args.output_dir + f"/dummy_{prefix}_token.json", "w", encoding="utf-8") as fout:
        json.dump(dummy_tokens, fout, indent=4)


def smart_emb_init(tokenizer, model):
    print("Initializing AMR Vocab according to similar tokens ...")
    for tok, idx in tokenizer.encoder.items():
        tok = tok.lstrip(tokenizer.INIT)

        if idx < tokenizer.old_enc_size:
            continue

        elif tok.startswith("<pointer:") and tok.endswith(">"):
            tok_split = ["pointer", str(tok.split(":")[1].strip(">"))]

        elif tok.startswith("<"):
            continue

        elif tok.startswith(":"):

            if tok.startswith(":op"):
                tok_split = ["relation", "operator", str(int(tok[3:]))]

            elif tok.startswith(":snt"):
                tok_split = ["relation", "sentence", str(int(tok[4:]))]

            elif tok.startswith(":ARG"):
                tok_split = ["relation", "argument", str(int(tok[4:]))]

            else:
                tok_split = ["relation"] + tok.lstrip(":").split("-")

        else:
            tok_split = tok.split("-")

        tok_split_ = tok_split
        tok_split = []
        for s in tok_split_:
            s_ = s + tokenizer.INIT
            if s_ in tokenizer.encoder:
                # print(f"{s_} in tokenizer vocabulary")
                tok_split.append(s_)
            else:
                tok_split.extend(tokenizer._tok_bpe(s))  #

        vecs = []
        for s in tok_split:
            idx_split = tokenizer.encoder.get(s, -1)
            if idx_split > -1:
                vec_split = model.model.shared.weight.data[idx_split].clone()
                vecs.append(vec_split)

        if vecs:
            vec = torch.stack(vecs, 0).mean(0)
            noise = torch.empty_like(vec)
            noise.uniform_(-0.1, +0.1)
            model.model.shared.weight.data[idx] = vec + noise

    return model


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="The model architecture to be trained or fine-tuned.",
    )
    # Other parameters
    parser.add_argument(
        "--val_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue",
        action="store_true",
        help="Whether to continue from latest checkpoint in output_dir",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm",
        action="store_true",
        help="Train with masked-language modeling loss instead of language modeling.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=1.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--joint_train_interval",
        default=1,
        type=int,
        help="The interval of joint AMR and text training",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument(
        "--smart_init",
        action="store_true",
        help="Whether to apply smart initialization to new token embedddings",
    )
    parser.add_argument(
        "--mlm_amr", action="store_true", help="Whether to apply mask language modeling on amrs",
    )
    parser.add_argument(
        "--mlm_amr_short",
        action="store_true",
        help="Whether to apply mask language modeling on amrs, short dec sequence",
    )
    parser.add_argument(
        "--mlm_text", action="store_true", help="Whether to apply mask language modeling on text",
    )
    parser.add_argument(
        "--mlm_text_short",
        action="store_true",
        help="Whether to apply mask language modeling on text, short dec sequence",
    )
    parser.add_argument(
        "--mlm_amr_plus_text",
        action="store_true",
        help="Whether to apply mask amr, plus text, short dec sequence",
    )
    parser.add_argument(
        "--mlm_text_plus_amr",
        action="store_true",
        help="Whether to apply mask text, plus amr, short dec sequence",
    )
    parser.add_argument(
        "--mlm_amr_plus_text_short",
        action="store_true",
        help="Whether to apply mask amr, plus text, short dec sequence",
    )
    parser.add_argument(
        "--mlm_text_plus_amr_short",
        action="store_true",
        help="Whether to apply mask text, plus amr, short dec sequence",
    )
    parser.add_argument(
        "--mlm_joint_to_amr", action="store_true", help="Whether to apply mask text, amr, to amr",
    )
    parser.add_argument(
        "--mlm_joint_to_text", action="store_true", help="Whether to apply mask text, amr, to text",
    )
    parser.add_argument(
        "--mlm_joint_to_joint",
        action="store_true",
        help="Whether to apply mask text, amr, to text amr",
    )
    parser.add_argument(
        "--freeze_embeds", action="store_true", help="Whether to freeze embeddings of the model",
    )
    parser.add_argument(
        "--freeze_encoder", action="store_true", help="Whether to freeze encoder of the model",
    )
    parser.add_argument(
        "--freeze_decoder", action="store_true", help="Whether to freeze decoder of the modele",
    )

    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    # if args.eval_data_file is None and args.do_eval:
    #     raise ValueError(
    #         "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
    #         "or remove the --do_eval argument."
    #     )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.should_continue
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    tokenizer = PENMANBartTokenizer.from_pretrained(
        args.model_name_or_path, collapse_name_ops=False, use_pointer_tokens=True, raw_graph=False,
    )

    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)

    if args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    if args.freeze_decoder:
        freeze_params(model.get_decoder())
        assert_all_frozen(model.get_decoder())

    if args.freeze_embeds:
        freeze_embeds(model)
    else:
        activate_embeds(model)

    model.to(args.device)
    print(model)
    train_p = [
        n for n, p in model.named_parameters() if p.requires_grad
    ]  # get the trainable params
    print(f"Trainable params in Summarization Model : {train_p}")

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    AMRDataset = AMRDataSetFast(
        tokenizer=tokenizer,
        train_file=args.train_file,
        validation_file=args.val_file,
        test_file=args.test_file,
        pad_to_max_length=False,
        max_src_length=args.block_size,
        max_tgt_length=256,
    )
    AMRDataset.setup()

    # Dummy Test
    train_dataset = AMRDataset.train_dataset
    dev_dataset = AMRDataset.valid_dataset

    seq2seq_collate_fn = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        if args.local_rank == 0:
            torch.distributed.barrier()

        if args.smart_init:
            model = smart_emb_init(tokenizer, model)

        global_step, tr_loss = train(
            args, train_dataset, dev_dataset, seq2seq_collate_fn, model, tokenizer, config
        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForMaskedLM.from_pretrained(args.output_dir)
        tokenizer = PENMANBartTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelForMaskedLM.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(
                args,
                dev_dataset,
                seq2seq_collate_fn,
                model,
                tokenizer,
                config=config,
                prefix=prefix,
            )
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
