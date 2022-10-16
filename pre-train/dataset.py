# coding:utf-8
import linecache
import os
import torch
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

# import git
from torch.utils.data import Dataset
from datasets import load_dataset
from dataclasses import dataclass
from transformers.file_utils import cached_property
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import re

nltk = None


def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    assert nltk, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


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


class AMRDataset(Dataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        tgt_pad_token_id=-100,
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".src")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".tgt")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")

        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]

        self.pad_token_id = self.tokenizer.pad_token_id
        self.tgt_pad_token_id = tgt_pad_token_id

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        src_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")

        assert src_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        source_inputs = self.tokenizer(
            src_line,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            target_inputs = self.tokenizer(
                tgt_line,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        # print('\ntarget input:', target_inputs)
        if self.pad_token_id != self.tgt_pad_token_id:
            target_inputs["input_ids"].masked_fill_(
                target_inputs["input_ids"] == self.pad_token_id, self.tgt_pad_token_id
            )

        source_ids = source_inputs["input_ids"]
        target_ids = target_inputs["input_ids"][1:]  # 去掉开头的<s>
        src_mask = source_inputs["attention_mask"]

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        y = trim_batch(target_ids, self.tgt_pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
        }
        return batch


class AMRDataSetFast(torch.nn.Module):
    def __init__(
        self,
        tokenizer,
        train_file,
        validation_file,
        test_file,
        prefix="",
        pad_to_max_length=True,
        max_src_length=512,
        max_tgt_length=512,
        ignore_pad_token_for_loss=True,
    ):
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.pad_to_max_length = pad_to_max_length
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length

    def amr_batch_encode(self, input_lst, max_length, pad_to_max_length=False):
        res = []
        for itm_lst in input_lst:
            res.append(
                self.get_ids(itm_lst, max_length=max_length, pad_to_max_length=pad_to_max_length)
            )
        return res
        # raw_data = torch.stack(res, dim=0)
        # keep_column_mask = raw_data.ne(self.tokenizer.pad_token_id).any(dim=0)
        # return raw_data[:, keep_column_mask]
        # return raw_data.tolist()

    def get_ids(self, tokens, max_length=0, pad_to_max_length=False):
        token_ids = [self.tokenizer.encoder.get(b, self.tokenizer.unk_token_id) for b in tokens]
        if pad_to_max_length:
            assert max_length > 0, "Invalid max-length: {}".format(max_length)
            pad_ids = [self.tokenizer.pad_token_id for _ in range(max_length)]
            len_tok = len(token_ids)
            if max_length > len_tok:
                pad_ids[:len_tok] = map(int, token_ids)
            else:
                pad_ids = token_ids[:max_length]
            return torch.tensor(pad_ids, dtype=torch.long)
        # return torch.tensor(token_ids, dtype=torch.long)
        return token_ids

    def setup(self, stage="fit"):
        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        data_files["test"] = self.test_file

        datasets = load_dataset("amrdata.py", data_files=data_files)
        print("datasets:", datasets)
        column_names = datasets["train"].column_names
        print("colums:", column_names)
        padding = "max_length" if self.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            src = examples["src"]  # text tokens
            tgt = examples["tgt"]  # amr tokens
            src = [self.prefix + inp for inp in src]
            tgt = [inp.split() + [self.tokenizer.amr_eos_token] for inp in tgt]

            model_inputs = self.tokenizer(
                src, max_length=self.max_src_length, padding=False, truncation=True
            )
            # Setup the tokenizer for targets
            # with self.tokenizer.as_target_tokenizer():
            #     labels = self.tokenizer(tgt, max_length=self.max_tgt_length, padding=False, truncation=True)
            tgt_ids = self.amr_batch_encode(
                tgt, max_length=self.max_tgt_length, pad_to_max_length=False
            )
            model_inputs["labels"] = tgt_ids
            joint_ids = [
                srci + [self.tokenizer.amr_bos_token_id] + tgti
                for srci, tgti in zip(model_inputs["input_ids"], model_inputs["labels"])
            ]  # [<s> x1,x2...,xn <\s> y1,y2,...ym <\s>]
            max_src_length = min(self.max_src_length * 2, 512)
            joint_ids = [
                itm[: max_src_length - 1] + [self.tokenizer.amr_eos_token_id]
                if len(itm) > max_src_length
                else itm
                for itm in joint_ids
            ]
            seg_ids = [
                [0 for _ in range(len(srci))] + [1 for _ in range(len(tgti) + 1)]
                for srci, tgti in zip(model_inputs["input_ids"], model_inputs["labels"])
            ]  # [<s> x1,x2...,xn <\s> y1,y2,...ym <\s>]
            seg_ids = [itm[: max_src_length] for itm in seg_ids]
            model_inputs["joint_ids"] = joint_ids
            model_inputs["seg_ids"] = seg_ids
            srcEtgt_ids = [
                srci[: self.max_src_length - 4]
                + [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.amr_bos_token_id,
                    self.tokenizer.mask_token_id,
                    self.tokenizer.amr_eos_token_id,
                ]
                if len(srci) > self.max_src_length - 3
                else srci
                + [
                    self.tokenizer.amr_bos_token_id,
                    self.tokenizer.mask_token_id,
                    self.tokenizer.amr_eos_token_id,
                ]
                for srci in model_inputs["input_ids"]
            ]  # [<s> x1,x2...,xn <\s> <AMR> [mask] <\s>]
            Esrctgt_ids = [
                [
                    self.tokenizer.bos_token_id,
                    self.tokenizer.mask_token_id,
                    self.tokenizer.eos_token_id,
                    self.tokenizer.amr_bos_token_id
                ]
                + tgti
                if len(tgti) <= self.max_src_length - 4
                else
                [
                    self.tokenizer.bos_token_id,
                    self.tokenizer.mask_token_id,
                    self.tokenizer.eos_token_id,
                    self.tokenizer.amr_bos_token_id
                ]
                + tgti[: self.max_src_length - 5]
                + [self.tokenizer.amr_eos_token_id]
                for tgti in model_inputs["labels"]
            ]  # [<s> [mask] <\s> <AMR> y1,y2...,yn <\s>]

            Esrctgt_segids = [
                [0 for _ in range(3)] + [1 for _ in range(len(itm) - 3)]
                for itm in Esrctgt_ids
            ]
            srcEtgt_segids = [
                [0 for _ in range(len(itm) - 3)] + [1 for _ in range(3)]
                for itm in srcEtgt_ids
            ]
            model_inputs["srcEtgt_ids"] = srcEtgt_ids
            model_inputs["srcEtgt_segids"] = srcEtgt_segids
            model_inputs["Esrctgt_ids"] = Esrctgt_ids
            model_inputs["Esrctgt_segids"] = Esrctgt_segids
            return model_inputs

        self.train_dataset = datasets["train"].map(
            tokenize_function, batched=True, remove_columns=["src", "tgt"], num_proc=8
        )
        print(f"ALL {len(self.train_dataset)} training instances")
        self.valid_dataset = datasets["validation"].map(
            tokenize_function, batched=True, remove_columns=["src", "tgt"], num_proc=8
        )
        print(f"ALL {len(self.valid_dataset)} validation instances")

        self.test_dataset = datasets["test"].map(
            tokenize_function, batched=True, remove_columns=["src", "tgt"], num_proc=8
        )
        print(f"ALL {len(self.test_dataset)} test instances")

        print("Dataset Instance Example:", self.train_dataset[0])


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
    label_pad_token_id: int = -100

    def __call__(self, features):
        # print("Features:", features)
        # labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # # same length to return tensors.
        # if labels is not None:
        #     max_label_length = max(len(l) for l in labels)
        #     padding_side = self.tokenizer.padding_side
        #     for feature in features:
        #         remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
        #         feature["labels"] = (
        #             feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
        #         )
        # print("Features:", features)
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.label_pad_token_id,
            key="labels",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="joint_ids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="seg_ids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="srcEtgt_ids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="srcEtgt_segids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="Esrctgt_ids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="Esrctgt_segids",
        )
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        # decoder_input_ids = shift_tokens_right(
        #     features["labels"],
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     decoder_start_token_id=self.tokenizer.amr_bos_token_id,
        # )

        return {
            "input_ids": features["input_ids"],
            "labels": features["labels"],
            "joint_ids": features["joint_ids"],
            "seg_ids": features["seg_ids"],
            "srcEtgt_ids": features["srcEtgt_ids"],
            "srcEtgt_segids": features["srcEtgt_segids"],
            "Esrctgt_ids": features["Esrctgt_ids"],
            "Esrctgt_segids": features["Esrctgt_segids"],
        }
