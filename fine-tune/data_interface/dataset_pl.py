# coding:utf-8
import os
import torch
import inspect
import importlib
import pytorch_lightning as pl
from datasets import load_dataset
from dataclasses import dataclass
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Optional, Union
from torch.utils.data.dataloader import DataLoader
from common.utils import shift_tokens_right


class AMR2TextDataModule(pl.LightningDataModule):
    def __init__(
        self, tokenizer, **args,
    ):
        super().__init__()
        self.train_file = args["train_data_file"]
        self.validation_file = args["eval_data_file"]
        self.test_file = args["test_data_file"]
        self.src_prefix = args["src_prefix"]
        self.tgt_prefix = args["tgt_prefix"]
        self.pad_to_max_length = False
        self.ignore_pad_token_for_loss = True
        self.cache_dir = args["cache_dir"]
        self.unified_inp = args["unified_input"]
        self.train_batchsize = args["per_gpu_train_batch_size"]
        self.val_batchsize = args["per_gpu_eval_batch_size"]
        self.train_num_worker = args["train_num_workers"]
        self.val_num_worker = args["eval_num_workers"]
        self.preprocess_worker = args["process_num_workers"]

        self.tokenizer = tokenizer
        print("Tokenizer:", len(self.tokenizer), self.tokenizer)
        self.max_src_length = min(args["src_block_size"], self.tokenizer.model_max_length)
        self.max_tgt_length = min(args["tgt_block_size"], self.tokenizer.model_max_length)

        self.collate_fn = DataCollatorForSeq2Seq(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if args["fp16"] else None,
            decoder_start_token_id=self.tokenizer.bos_token_id,
        )

    def setup(self, stage="fit"):
        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        data_files["test"] = self.test_file

        # print("datafiles:", data_files)
        print("Dataset cache dir:", self.cache_dir)
        # print(f"{os.path.dirname(__file__)}/amrdata.py")
        # exit()
        datasets = load_dataset(
            f"{os.path.dirname(__file__)}/data.py", data_files=data_files, cache_dir=self.cache_dir,
        )
        print("datasets:", datasets)
        column_names = datasets["train"].column_names
        print("colums:", column_names)

        def tokenize_function(examples):
            # Remove empty lines
            sents = examples["src"]  # text tokens
            amrs = examples["tgt"]  # amr tokens
            model_inputs = {}
            if self.unified_inp:
                amr_tokens = [
                    [self.tokenizer.bos_token, self.tokenizer.mask_token, self.tokenizer.eos_token]
                    + [self.tokenizer.amr_bos_token]
                    + amr.split()
                    + [self.tokenizer.amr_eos_token]
                    for amr in amrs
                ]
            else:
                amr_tokens = [
                    [self.tokenizer.bos_token] + amr.split() + [self.tokenizer.eos_token]
                    for amr in amrs
                ]
            src_ids = amr_batch_encode(
                amr_tokens, max_length=self.max_src_length, pad_to_max_length=False
            )
            # src_mask = src_ids.ne(self.tokenizer.pad_token_id).int()
            tgt_ids = self.tokenizer.batch_encode_plus(
                sents, max_length=self.max_tgt_length, padding=False, truncation=True
            )["input_ids"]
            label_ids = [[l for l in label[1:]] for label in tgt_ids]
            model_inputs["input_ids"] = src_ids
            model_inputs["labels"] = label_ids
            return model_inputs

        def amr_batch_encode(input_lst, max_length, pad_to_max_length=False):
            res = []
            for itm_lst in input_lst:
                res.append(
                    get_ids(itm_lst, max_length=max_length, pad_to_max_length=pad_to_max_length)
                )
            return res

        def get_ids(tokens, max_length=0, pad_to_max_length=False):
            token_ids = [self.tokenizer.encoder.get(b, self.tokenizer.unk_token_id) for b in tokens]
            if pad_to_max_length:
                assert max_length > 0, "Invalid max-length: {}".format(max_length)
                pad_ids = [self.tokenizer.pad_token_id for _ in range(max_length)]
                len_tok = len(token_ids)
                if max_length > len_tok:
                    pad_ids[:len_tok] = map(int, token_ids)
                else:
                    pad_ids = token_ids[:max_length]
                return pad_ids
            return token_ids

        self.train_dataset = datasets["train"].map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocess_worker,
            remove_columns=["src", "tgt"],
        )
        print(f"ALL {len(self.train_dataset)} training instances")
        self.valid_dataset = datasets["validation"].map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocess_worker,
            remove_columns=["src", "tgt"],
        )
        print(f"ALL {len(self.valid_dataset)} validation instances")

        self.test_dataset = datasets["test"].map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocess_worker,
            remove_columns=["src", "tgt"],
        )
        print(f"ALL {len(self.test_dataset)} test instances")
        print("Dataset Instance Example:", self.train_dataset[0])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batchsize,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.train_num_worker,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.val_batchsize,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.val_num_worker,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batchsize,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.val_num_worker,
            pin_memory=True,
        )

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            self.data_module = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except:
            raise ValueError(
                f"Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}"
            )

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)


class AMRParsingDataModule(pl.LightningDataModule):
    def __init__(
        self, tokenizer, **args,
    ):
        super().__init__()
        self.train_file = args["train_data_file"]
        self.validation_file = args["eval_data_file"]
        self.test_file = args["test_data_file"]
        self.src_prefix = args["src_prefix"]
        self.tgt_prefix = args["tgt_prefix"]
        self.pad_to_max_length = False
        self.ignore_pad_token_for_loss = True
        self.cache_dir = args["cache_dir"]
        self.unified_inp = args["unified_input"]
        self.train_batchsize = args["per_gpu_train_batch_size"]
        self.val_batchsize = args["per_gpu_eval_batch_size"]
        self.train_num_worker = args["train_num_workers"]
        self.val_num_worker = args["eval_num_workers"]
        self.preprocess_worker = args["process_num_workers"]

        self.tokenizer = tokenizer
        print("Tokenizer:", len(self.tokenizer), self.tokenizer)
        self.max_src_length = min(args["src_block_size"], self.tokenizer.model_max_length)
        self.max_tgt_length = min(args["tgt_block_size"], self.tokenizer.model_max_length)

        self.collate_fn = DataCollatorForSeq2Seq(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if args["fp16"] else None,
            decoder_start_token_id=self.tokenizer.amr_bos_token_id,
        )

    def setup(self, stage="fit"):
        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        data_files["test"] = self.test_file

        # print("datafiles:", data_files)
        print("Dataset cache dir:", self.cache_dir)
        # print(f"{os.path.dirname(__file__)}/amrdata.py")
        # exit()
        datasets = load_dataset(
            f"{os.path.dirname(__file__)}/data.py", data_files=data_files, cache_dir=self.cache_dir,
        )
        print("datasets:", datasets)
        column_names = datasets["train"].column_names
        print("colums:", column_names)

        def tokenize_function(examples):
            # Remove empty lines
            sents = examples["src"]  # text tokens
            amrs = examples["tgt"]  # amr tokens
            model_inputs = {}
            src_ids = self.tokenizer.batch_encode_plus(
                sents, max_length=self.max_tgt_length, padding=False, truncation=True
            )["input_ids"]

            if self.unified_inp:
                src_ids = [itm + [self.tokenizer.amr_bos_token_id, self.tokenizer.mask_token_id, self.tokenizer.amr_eos_token_id] for itm in src_ids]

            amr_tokens = [
                [self.tokenizer.amr_bos_token] + amr.split() + [self.tokenizer.amr_eos_token]
                for amr in amrs
            ]
            tgt_ids = amr_batch_encode(
                amr_tokens, max_length=self.max_src_length, pad_to_max_length=False
            )
            label_ids = [[l for l in label[1:]] for label in tgt_ids]
            model_inputs["input_ids"] = src_ids
            model_inputs["labels"] = label_ids
            return model_inputs

        def amr_batch_encode(input_lst, max_length, pad_to_max_length=False):
            res = []
            for itm_lst in input_lst:
                res.append(
                    get_ids(itm_lst, max_length=max_length, pad_to_max_length=pad_to_max_length)
                )
            return res

        def get_ids(tokens, max_length=0, pad_to_max_length=False):
            token_ids = [self.tokenizer.encoder.get(b, self.tokenizer.unk_token_id) for b in tokens]
            if pad_to_max_length:
                assert max_length > 0, "Invalid max-length: {}".format(max_length)
                pad_ids = [self.tokenizer.pad_token_id for _ in range(max_length)]
                len_tok = len(token_ids)
                if max_length > len_tok:
                    pad_ids[:len_tok] = map(int, token_ids)
                else:
                    pad_ids = token_ids[:max_length]
                return pad_ids
            return token_ids

        self.train_dataset = datasets["train"].map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocess_worker,
            remove_columns=["src", "tgt"],
        )
        print(f"ALL {len(self.train_dataset)} training instances")

        self.valid_dataset = datasets["validation"].map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocess_worker,
            remove_columns=["src", "tgt"],
        )
        print(f"ALL {len(self.valid_dataset)} validation instances")

        self.test_dataset = datasets["test"].map(
            tokenize_function,
            batched=True,
            num_proc=self.preprocess_worker,
            remove_columns=["src", "tgt"],
        )
        print(f"ALL {len(self.test_dataset)} test instances")
        print("Dataset Instance Example:", self.train_dataset[0])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batchsize,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.train_num_worker,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.val_batchsize,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.val_num_worker,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batchsize,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.val_num_worker,
            pin_memory=True,
        )

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            self.data_module = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except:
            raise ValueError(
                f"Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}"
            )

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)


def padding_func(features, padding_side="right", pad_token_id=1, key="label"):
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features)
    for feature in features:
        remainder = [pad_token_id] * (max_label_length - len(feature[key]))
        feature[key] = (
            feature[key] + remainder if padding_side == "right" else remainder + feature[key]
        )
    return


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    decoder_start_token_id: int = 0
    label_pad_token_id: int = -100

    def __call__(self, features):
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.label_pad_token_id,
            key="labels",
        )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        attention_mask = features["input_ids"].ne(self.tokenizer.pad_token_id).int()
        # prepare decoder_input_ids

        features["decoder_input_ids"] = shift_tokens_right(
            features["labels"],
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

        return {
            "input_ids": features["input_ids"],
            "attention_mask": attention_mask,
            "labels": features["labels"],
            "decoder_input_ids": features["decoder_input_ids"],
        }
