# coding=utf-8

import os
import re
import sys
from textwrap import indent
import json
import nltk  # Here to have a nice missing dependency error message early on
import torch
import penman
import logging
import datasets
import transformers
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, load_metric, load_from_disk
from data_interface.dataset import AMR2TextDataSet, AMRParsingDataSet, DataCollatorForAMR2Text, DataCollatorForAMRParsing
from model_interface.modeling_bart import BartForConditionalGeneration
from model_interface.tokenization_bart import AMRBartTokenizer
from common.options import DataTrainingArguments, ModelArguments, Seq2SeqTrainingArguments
from common.utils import smart_emb_init, calculate_smatch
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartTokenizer,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
from seq2seq_trainer import Seq2SeqTrainer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.21.0.dev0")

require_version(
    "datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt"
)

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
]


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert training_args.task in ("amr2text", "text2amr"), f"Invalid task name:{training_args.task}, should be in ['amr2text', 'text2amr')"
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    tokenizer = AMRBartTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = BartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    # config dec_start_token, max_pos_embeddings
    if model.config.decoder_start_token_id is None and isinstance(
        tokenizer, (MBartTokenizer, MBartTokenizerFast)
    ):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )
            
    if training_args.do_train and training_args.smart_init and training_args.resume_from_checkpoint is None and last_checkpoint is None:
        smart_emb_init(tokenizer, model)
            
    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    DataSetCate = AMR2TextDataSet if training_args.task == "amr2text" else AMRParsingDataSet
    raw_datasets = DataSetCate(tokenizer, data_args, model_args)
    
    column_names = raw_datasets.datasets["train"].column_names

    if training_args.do_train:
        if "train" not in raw_datasets.datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets.datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        
        if data_args.overwrite_cache or not os.path.exists(data_args.data_cache_dir + "/train"):
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    raw_datasets.tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
                print("Saving cached train data ...")
                train_dataset.save_to_disk(data_args.data_cache_dir + "/train")
        else:
            train_dataset = load_from_disk(data_args.data_cache_dir + "/train", keep_in_memory=True)

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets.datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets.datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        if data_args.overwrite_cache or not os.path.exists(data_args.data_cache_dir + "/valid"):
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_dataset.map(
                    raw_datasets.tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )
                print("Saving cached validation data ...")
                eval_dataset.save_to_disk(data_args.data_cache_dir + "/valid")
        else:
            eval_dataset = load_from_disk(data_args.data_cache_dir + "/valid", keep_in_memory=True)

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets.datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets.datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        if data_args.overwrite_cache or not os.path.exists(data_args.data_cache_dir + "/test"):
            with training_args.main_process_first(desc="prediction dataset map pre-processing"):
                predict_dataset = predict_dataset.map(
                    raw_datasets.tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )
                print("Saving cached test data ...")
                predict_dataset.save_to_disk(data_args.data_cache_dir + "/test")
        else:
            predict_dataset = load_from_disk(data_args.data_cache_dir + "/test", keep_in_memory=True)

    # label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    label_pad_token_id = tokenizer.pad_token_id
    
    DataCollatorCate = DataCollatorForAMR2Text if training_args.task == "amr2text" else DataCollatorForAMRParsing
    data_collator = DataCollatorCate(
        tokenizer,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    metric = load_metric(path="metric/sacrebleu.py") if training_args.task == "amr2text" else None

    def compute_metrics_parsing(eval_preds, global_step=0, prefix="val"):
        prefix = "test" if prefix == "predict" else "val"
        preds, labels, inputs = eval_preds
        # print("inputs:", inputs)
        # print("Pred", preds)
        # print("labels", labels)
        if isinstance(preds, tuple):
            preds = preds[0]
        
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)  
        # if data_args.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        os.makedirs(training_args.output_dir + "/val_outputs/", exist_ok=True)
        
        # gen_graphs = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # debug_output = f"{training_args.output_dir}/val_outputs/{prefix}_nodes_{global_step}.json"
        
        # with open(debug_output, 'w', encoding='utf-8') as fout:
        #     json.dump(gen_graphs, fout, indent=4)
        
        graphs = []
        for idx in range(len(preds)):
            graphs_same_source = []
            graphs.append(graphs_same_source)
            ith_pred = preds[idx]
            ith_pred[0] = tokenizer.bos_token_id
            ith_pred = [
                tokenizer.eos_token_id if itm == tokenizer.amr_eos_token_id else itm
                for itm in ith_pred if itm != tokenizer.pad_token_id
            ]
            
            graph, status, (lin, backr) = tokenizer.decode_amr(
                ith_pred, restore_name_ops=False
            )
            graph.status = status
            graph.nodes = lin
            graph.backreferences = backr
            graph.tokens = ith_pred
            graphs_same_source.append(graph)
        
        graphs_same_source[:] = tuple(
            zip(*sorted(enumerate(graphs_same_source), key=lambda x: (x[1].status.value, x[0])))
        )[1]
        idx = 0
        assert len(graphs) == len(decoded_inputs), f"inconsistent lenths {len(graphs)} vs {len(decoded_inputs)}"
        for gps, snt in zip(graphs, decoded_inputs):
            # print(gps, src)
            # exit()
            for gp in gps:
                # metadata = gg.metadata.copy()
                metadata = {}
                metadata["id"] = str(idx)
                metadata["annotator"] = "bart-amr"
                # metadata["date"] = str(datetime.datetime.now())
                metadata["snt"] = snt.replace("<AMR>", '').replace("</AMR>", '').strip()
                if "save-date" in metadata:
                    del metadata["save-date"]
                gp.metadata = metadata
                idx += 1
        
        # print("Before Penman Encoding")
        pieces = [penman.encode(g[0]) for g in graphs]
        output_prediction_file = f"{training_args.output_dir}/val_outputs/{prefix}_generated_predictions_{global_step}.txt"
        # write predictions and targets for later rouge evaluation.
        with open(output_prediction_file, "w") as p_writer:
            p_writer.write("\n\n".join(pieces))
        try:
            smatch_score = calculate_smatch(
                data_args.data_dir + f"/{prefix}-gold.amr", output_prediction_file
            )
        except:
            smatch_score = {"smatch": 0.0}
            
        result = {"smatch":smatch_score["smatch"]}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    def compute_metrics_generation(eval_preds, global_step=0, prefix="val"):
        prefix = "test" if prefix == "predict" else "val"
        preds, labels, inputs = eval_preds
        # print("inputs:", inputs)
        # print("Pred", preds)
        # print("labels", labels)
        if isinstance(preds, tuple):
            preds = preds[0]
        
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [[label.strip()] for label in labels]           # sacrebleu uses multi reference setting
            return preds, labels
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, lowercase=True
        )
        print(result)
        result = {"bleu": result["score"]}
        os.makedirs(training_args.output_dir + "/val_outputs/", exist_ok=True)
        output_prediction_file = f"{training_args.output_dir}/val_outputs/{prefix}_generated_predictions_{global_step}.txt"
        with open(output_prediction_file, "w") as p_writer:
            p_writer.write("\n".join(decoded_preds) + "\n")
            
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    es_callback = EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping)
    training_args.max_target_length = data_args.max_target_length
    
    compute_metrics = compute_metrics_generation if training_args.task == "amr2text" else compute_metrics_parsing
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[es_callback],
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )

    num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )

    if training_args.do_eval:
        
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.txt"
                )
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
