# coding:utf-8

import math
import torch
import numpy as np
from torch import nn
from typing import Callable, Dict, Iterable, List, Tuple, Union, Any
from torch.optim import Optimizer
from numpy.lib.twodim_base import mask_indices
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
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

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def freeze_params_amr(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for name, par in model.named_parameters():
        if 'adapter' not in name:
            par.requires_grad = False

def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def act_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = True


def activate_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        act_params(model.shared)
        for d in [model.encoder, model.decoder]:
            act_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            act_params(d.embed_positions)
            act_params(d.embed_tokens)
    else:
        act_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            act_params(d.embed_positions)
            act_params(d.embed_tokens)


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def get_inverse_sqrt_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, (num_warmup_steps / current_step)**0.5)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def mask_span(tokenizer, inputs: Any, plm_probability: float = 1 / 6, max_span_length: int = 5) -> Tuple[Any, Any, Any, Any]:
    """
    The masked tokens to be predicted for a particular sequence are determined by the following algorithm:
        0. Start from the beginning of the sequence by setting ``cur_len = 0`` (number of tokens processed so far).
        1. Sample a ``span_length`` from the interval ``[1, max_span_length]`` (length of span of tokens to be
            masked)
        2. Reserve a context of length ``context_length = span_length / plm_probability`` to surround span to be
            masked
        3. Sample a starting point ``start_index`` from the interval ``[cur_len, cur_len + context_length -
            span_length]`` and mask tokens ``start_index:start_index + span_length``
        4. Set ``cur_len = cur_len + context_length``. If ``cur_len < max_len`` (i.e. there are tokens remaining in
            the sequence to be processed), repeat from Step 1.
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for permutation language modeling. Please add a mask token if you want to use this tokenizer."
        )

    labels = inputs.clone()
    masked_inputs = inputs.clone()
    # Creating the mask and target_mapping tensors
    masked_indices = torch.full(labels.shape, 0, dtype=torch.bool)

    for i in range(labels.size(0)):
        # Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
        cur_len = 0
        max_len = labels.size(1)

        while cur_len < max_len:
            # Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
            span_length = torch.randint(1, max_span_length + 1, (1,)).item()
            # Reserve a context of length `context_length = span_length / plm_probability` to surround the span to be masked
            context_length = int(span_length / plm_probability)
            # Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length - span_length]` and mask tokens `start_index:start_index + span_length`
            start_index = cur_len + torch.randint(context_length - span_length + 1, (1,)).item()
            masked_indices[i, start_index: start_index + span_length] = 1
            # Set `cur_len = cur_len + context_length`
            cur_len += context_length

    special_tokens_mask = torch.tensor(
        [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ],
        dtype=torch.bool,
    )
    masked_indices.masked_fill_(special_tokens_mask, value=0.0)

    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        masked_indices.masked_fill_(padding_mask, value=0.0)

    masked_inputs[masked_indices] = tokenizer.mask_token_id
    return masked_inputs


def sentence_infilling(tokenizer, inp, mlm_prob=0.35):
    token_length = len([int(itm != tokenizer.pad_token_id) for itm in inp])
    masking_length = math.floor(token_length * mlm_prob)
    masked_length = 1
    masked_inputs = inp.clone().tolist()
    while masked_length < masking_length:
        span_length = min(math.floor(np.random.poisson(3, 1)), token_length - 1)
        start_index = math.floor(np.random.uniform(1, token_length - span_length, 1))
        masked_inputs = masked_inputs[:start_index] + [tokenizer.mask_token_id] + masked_inputs[start_index + span_length:]
        token_length -= span_length - 1
        masked_length += span_length
    return torch.LongTensor(masked_inputs)


def text_infilling(inp, tokenizer, mlm_prob=0.35):
    res = []
    for sents in inp:
        res.append(sentence_infilling(tokenizer, sents, mlm_prob=mlm_prob))
    return pad_sequence(res, batch_first=True, padding_value=tokenizer.pad_token_id)


def mask_tokens(
    inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    mask_inputs = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        special_token_mask = mask_inputs.eq(tokenizer.amr_bos_token_id)     # don't mask AMR_bos_token
        probability_matrix.masked_fill_(special_token_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100                      # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    mask_inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    mask_inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return mask_inputs, labels


def mask_tokens_short(
    inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    INIT = 'Ġ'
    labels = inputs.clone()
    mask_inputs = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        special_token_mask = mask_inputs.eq(tokenizer.amr_bos_token_id)         # don't mask AMR_bos_token
        probability_matrix.masked_fill_(special_token_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()                 # True/1 mask, False/0 dont mask
    labels[~masked_indices] = -100                      # We only compute loss on masked tokens

    mask_inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    res_labels = []
    for inp_itm, label_itm in zip(mask_inputs, labels):
        ith_label = []
        cnt = 1
        # for itm_idx, ith_label_idx in zip(inp_itm, label_itm):
        #     if itm_idx == tokenizer.mask_token_id:
        #         mask_id = tokenizer.convert_tokens_to_ids(f"{INIT}<mask{cnt}>")
        #         inp_itm[itm_idx] = mask_id
        #         ith_label.extend([mask_id, ith_label_idx])
        for iidx in range(len(inp_itm)):
            if inp_itm[iidx] == tokenizer.mask_token_id:
                mask_id = tokenizer.convert_tokens_to_ids(f"{INIT}<mask{cnt}>")
                inp_itm[iidx] = mask_id
                ith_label.extend([mask_id, label_itm[iidx]])
                cnt += 1

        ith_label.append(tokenizer.eos_token_id)
        res_labels.append(ith_label)
    
    labels = torch.tensor(tokenizer.pad({"input_ids": res_labels})["input_ids"])
    labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return mask_inputs, labels


def mask_joint_tokens_short(
    inputs: torch.Tensor, seg_ids: torch.Tensor, tokenizer: PreTrainedTokenizer, args, mask_txt=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    INIT = 'Ġ'
    labels = inputs.clone()
    mask_inputs = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        special_token_mask = mask_inputs.eq(tokenizer.amr_bos_token_id)     # don't mask AMR_bos_token
        probability_matrix.masked_fill_(special_token_mask, value=0.0)

    assert seg_ids.size() == inputs.size(), f"inconsistent size between input and seg_ids: {str(inputs.size())}, {str(seg_ids.size())}"
    
    if mask_txt:
        probability_matrix.masked_fill_(seg_ids == 1, value=0.0)      # if mask text, set seg1 to 0
    else:
        probability_matrix.masked_fill_(seg_ids == 0, value=0.0)      # if mask AMR, set seg0 to 0

    masked_indices = torch.bernoulli(probability_matrix).bool()     # True/1 mask, False/0 dont mask
    labels[~masked_indices] = -100                      # We only compute loss on masked tokens

    mask_inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    res_labels = []
    for inp_itm, label_itm in zip(mask_inputs, labels):
        ith_label = []
        cnt = 1
        # for itm_idx, ith_label_idx in zip(inp_itm, label_itm):
        #     if itm_idx == tokenizer.mask_token_id:
        #         mask_id = tokenizer.convert_tokens_to_ids(f"{INIT}<mask{cnt}>")
        #         inp_itm[itm_idx] = mask_id
        #         ith_label.extend([mask_id, ith_label_idx])
        for iidx in range(len(inp_itm)):
            if inp_itm[iidx] == tokenizer.mask_token_id:
                mask_id = tokenizer.convert_tokens_to_ids(f"{INIT}<mask{cnt}>")
                inp_itm[iidx] = mask_id
                ith_label.extend([mask_id, label_itm[iidx]])
                cnt += 1

        ith_label.append(tokenizer.eos_token_id)
        res_labels.append(ith_label)
    
    labels = torch.tensor(tokenizer.pad({"input_ids": res_labels})["input_ids"])
    labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return mask_inputs, labels


def mask_joint_tokens_full(
    inputs: torch.Tensor, seg_ids: torch.Tensor, tokenizer: PreTrainedTokenizer, args, mask_txt=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    labels = inputs.clone()
    mask_inputs = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        special_token_mask = mask_inputs.eq(tokenizer.amr_bos_token_id)     # don't mask AMR_bos_token
        probability_matrix.masked_fill_(special_token_mask, value=0.0)
        
    assert seg_ids.size() == inputs.size(), f"inconsistent size between input and seg_ids: {str(inputs.size())}, {str(seg_ids.size())}"
    
    if mask_txt:
        probability_matrix.masked_fill_(seg_ids == 1, value=0.0)      # if mask text, set seg1 to 0
    else:
        probability_matrix.masked_fill_(seg_ids == 0, value=0.0)      # if mask AMR, set seg0 to 0

    masked_indices = torch.bernoulli(probability_matrix).bool()       # True/1 mask, False/0 dont mask
    mask_inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    return mask_inputs, None


def get_mlm_inputs(batch, tokenizer, args, inp='text'):
    if inp == "text":
        ori_input = batch["input_ids"]
        masked_input, _ = mask_tokens(ori_input, tokenizer, args) if args.mlm else (batch, batch)
        attention_mask = batch["attention_mask"]
        labels = ori_input.clone()
        labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
        labels = labels[:, 1:]                      # [w1 w2 w3 ...]
        dec_input = ori_input[:, :-1]
        # dec_input = dec_input.masked_fill_(dec_input == tokenizer.bos_token_id, tokenizer.text_bos_token_id)   # <ToText> w1, w2, ..., wn <\s>
        return masked_input, attention_mask, dec_input, labels
    
    elif inp == "amr":
        labels = batch["labels"]                                # [bsz, len+1]
        shifted_input_ids = labels.new_zeros(labels.size(0), labels.size(1) + 1)
        shifted_input_ids[:, 1:] = labels.clone()
        # shifted_input_ids[:, 0] = tokenizer.bos_token_id                    # <s> w1, w2, ..., wn <\s>
        shifted_input_ids[:, 0] = tokenizer.amr_bos_token_id                # <AMR> w1, w2, ..., wn <\s>
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, tokenizer.pad_token_id)   # replace -100 with pad_token_id
        masked_input, _ = mask_tokens(shifted_input_ids, tokenizer, args)
        attention_mask = shifted_input_ids.ne(tokenizer.pad_token_id).int()     # attention mask
        dec_input = batch["decoder_input_ids"]
        return masked_input, attention_mask, dec_input, labels


def get_text_infilling_inputs(batch, tokenizer, args, inp='text'):
    if inp == "text":
        ori_input = batch["input_ids"]
        masked_input = text_infilling(ori_input, tokenizer)
        attention_mask = masked_input.ne(tokenizer.pad_token_id).int()
        labels = ori_input.clone()
        labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
        labels = labels[:, 1:]                      # [w1 w2 w3 ...]
        dec_input = ori_input[:, :-1]
        return masked_input, attention_mask, dec_input, labels
    
    elif inp == "amr":
        labels = batch["labels"]                                # [bsz, len+1]
        shifted_input_ids = labels.new_zeros(labels.size(0), labels.size(1) + 1)
        shifted_input_ids[:, 1:] = labels.clone()
        # shifted_input_ids[:, 0] = tokenizer.bos_token_id                    # <s> w1, w2, ..., wn <\s>
        shifted_input_ids[:, 0] = tokenizer.amr_bos_token_id                # <AMR> w1, w2, ..., wn <\s>
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, tokenizer.pad_token_id)   # replace -100 with pad_token_id
        masked_input = text_infilling(shifted_input_ids, tokenizer)
        attention_mask = masked_input.ne(tokenizer.pad_token_id).int()     # attention mask
        dec_input = batch["decoder_input_ids"]
        return masked_input, attention_mask, dec_input, labels


def get_mlm_inputs_short(batch, tokenizer, args, inp='text'):
    if inp == "text":
        ori_input = batch["input_ids"]
        masked_input, labels_new = mask_tokens_short(ori_input, tokenizer, args)
        attention_mask = batch["attention_mask"]
        dec_input = labels_new.new_zeros(labels_new.size(0), labels_new.size(1))
        dec_input[:, 1:] = labels_new[:, :-1].clone()
        dec_input[:, 0] = tokenizer.bos_token_id                                # short use <s> as start token
        dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)
        return masked_input, attention_mask, dec_input, labels_new
    
    elif inp == "amr":
        amr = batch["labels"]                                # [bsz, len+1]
        input_ids = amr.new_zeros(amr.size(0), amr.size(1) + 1)
        input_ids[:, 1:] = amr.clone()
        # input_ids[:, 0] = tokenizer.bos_token_id                    # <s> w1, w2, ..., wn <\s>
        input_ids[:, 0] = tokenizer.amr_bos_token_id                # <AMR> w1, w2, ..., wn <\s>
        input_ids.masked_fill_(input_ids == -100, tokenizer.pad_token_id)   # replace -100 with pad_token_id
        masked_input, labels_new = mask_tokens_short(input_ids, tokenizer, args)
        attention_mask = masked_input.ne(tokenizer.pad_token_id).int()     # attention mask
        dec_input = labels_new.new_zeros(labels_new.size(0), labels_new.size(1))
        dec_input[:, 1:] = labels_new[:, :-1].clone()
        dec_input[:, 0] = tokenizer.bos_token_id                                # short use <s> as start token
        dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)
        return masked_input, attention_mask, dec_input, labels_new


def get_mlm_joint_inputs_short(batch, tokenizer, args, inp='text'):
    ori_input = batch["joint_ids"]
    seg_ids = batch["seg_ids"]
    if inp == 'text':
        masked_input, labels_new = mask_joint_tokens_short(ori_input, seg_ids, tokenizer, args, mask_txt=True)
    else:
        masked_input, labels_new = mask_joint_tokens_short(ori_input, seg_ids, tokenizer, args, mask_txt=False)

    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()      # attention mask
    dec_input = labels_new.new_zeros(labels_new.size(0), labels_new.size(1))
    dec_input[:, 1:] = labels_new[:, :-1].clone()
    dec_input[:, 0] = tokenizer.bos_token_id                            # short mlm use <s> as start token
    dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)
    return masked_input, attention_mask, dec_input, labels_new


def get_mlm_joint_inputs_full(batch, tokenizer, args, inp='text'):
    ori_input = batch["joint_ids"]
    seg_ids = batch["seg_ids"]
    if inp == 'text':
        masked_input, _ = mask_joint_tokens_full(ori_input, seg_ids, tokenizer, args, mask_txt=True)
        labels = batch["input_ids"].clone()
        labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
        labels = labels[:, 1:]                                                  # w1, w2, .., wn <\s>
        dec_input = labels.new_zeros(labels.size(0), labels.size(1))
        dec_input[:, 1:] = labels[:, :-1].clone()
        dec_input[:, 0] = tokenizer.bos_token_id                                # <s> w1 w2, ..., wn
        dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)
    else:
        masked_input, _ = mask_joint_tokens_full(ori_input, seg_ids, tokenizer, args, mask_txt=False)
        labels = batch["labels"]
        dec_input = batch["decoder_input_ids"]                                  # <AMR> w1 w2, ..., wn

    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()              # attention mask
    return masked_input, attention_mask, dec_input, labels


def get_partial_textinf_joint_inputs(batch, tokenizer, args, inp='text', mlm_prob=0.35):
    ori_input = batch["joint_ids"]
    seg_ids = batch["seg_ids"]
    if inp == 'text':
        masked_input = joint_infilling_partial(ori_input, seg_ids, tokenizer, mask_txt=True, mlm_prob=mlm_prob)
        labels = batch["input_ids"].clone()
        labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
        labels = labels[:, 1:]                                                  # w1, w2, .., wn <\s>
        dec_input = labels.new_zeros(labels.size(0), labels.size(1))
        dec_input[:, 1:] = labels[:, :-1].clone()
        dec_input[:, 0] = tokenizer.bos_token_id                                # <s> w1 w2, ..., wn
        dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)
    else:
        masked_input = joint_infilling_partial(ori_input, seg_ids, tokenizer, mask_txt=False, mlm_prob=mlm_prob)
        labels = batch["labels"]
        dec_input = batch["decoder_input_ids"]                                  # <AMR> w1 w2, ..., wn

    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()              # attention mask
    return masked_input, attention_mask, dec_input, labels


def joint_infilling_partial(inp, seg_id, tokenizer, mask_txt=True, mlm_prob=0.35):
    res = []
    for inp_ids, seg_iid in zip(inp, seg_id):
        inp_ids = torch.LongTensor([iid for iid in inp_ids if iid != tokenizer.pad_token_id])
        text_ids = torch.LongTensor([inp_ids[idx] for idx in range(len(inp_ids)) if seg_iid[idx] == 0])
        amr_ids = torch.LongTensor([inp_ids[idx] for idx in range(len(inp_ids)) if seg_iid[idx] == 1])
        if mask_txt:
            res.append(torch.cat([sentence_infilling(tokenizer, text_ids, mlm_prob=mlm_prob), amr_ids], dim=0))
        else:
            res.append(torch.cat([text_ids, sentence_infilling(tokenizer, amr_ids, mlm_prob=mlm_prob)], dim=0))

    return pad_sequence(res, batch_first=True, padding_value=tokenizer.pad_token_id)


def joint_infilling_full(inp, seg_id, tokenizer, mlm_prob=0.35):
    res = []
    for inp_ids, seg_iid in zip(inp, seg_id):
        inp_ids = torch.LongTensor([iid for iid in inp_ids if iid != tokenizer.pad_token_id])
        text_ids = torch.LongTensor([inp_ids[idx] for idx in range(len(inp_ids)) if seg_iid[idx] == 0])
        amr_ids = torch.LongTensor([inp_ids[idx] for idx in range(len(inp_ids)) if seg_iid[idx] == 1])
        masked_txt = sentence_infilling(tokenizer, text_ids, mlm_prob=mlm_prob)
        masked_amr = sentence_infilling(tokenizer, amr_ids, mlm_prob=mlm_prob)
        res.append(torch.cat([masked_txt, masked_amr], dim=0))

    return pad_sequence(res, batch_first=True, padding_value=tokenizer.pad_token_id)


def get_full_textinf_joint_inputs_partial_output(batch, tokenizer, args, inp='text', mlm_prob=0.35):
    ori_input = batch["joint_ids"]
    seg_ids = batch["seg_ids"]
    masked_input = joint_infilling_full(ori_input, seg_ids, tokenizer, mlm_prob=mlm_prob)
    if inp == 'text':
        labels = batch["input_ids"].clone()
        labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
        labels = labels[:, 1:]                                                  # w1, w2, .., wn <\s>
        dec_input = labels.new_zeros(labels.size(0), labels.size(1))
        dec_input[:, 1:] = labels[:, :-1].clone()
        dec_input[:, 0] = tokenizer.bos_token_id                                # <s> w1 w2, ..., wn
        dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)
    else:
        labels = batch["labels"]
        dec_input = batch["decoder_input_ids"]                                  # <AMR> w1 w2, ..., wn

    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()              # attention mask
    return masked_input, attention_mask, dec_input, labels


def get_full_textinf_joint_inputs_joint_output(batch, tokenizer, mlm_prob=0.35):
    ori_input = batch["joint_ids"]
    seg_ids = batch["seg_ids"]
    masked_input = joint_infilling_full(ori_input, seg_ids, tokenizer, mlm_prob=mlm_prob)
    
    labels = batch["joint_ids"].clone()
    labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
    labels = labels[:, 1:]                                                  # w1, w2, .., wn <\s>
    dec_input = labels.new_zeros(labels.size(0), labels.size(1))
    dec_input[:, 1:] = labels[:, :-1].clone()
    dec_input[:, 0] = tokenizer.bos_token_id                                # <s> w1 w2, ..., wn
    dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)

    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()             # attention mask
    return masked_input, attention_mask, dec_input, labels


def get_full_textinf_joint_inputs_partial_output_tapt(batch, tokenizer, mlm_prob=0.35):
    ori_input = batch["input_ids"]
    seg_ids = batch["seg_ids"]
    masked_input = joint_infilling_partial(ori_input, seg_ids, tokenizer, mask_txt=True, mlm_prob=mlm_prob)
    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()              # attention mask
    labels = batch["labels"]
    dec_input = labels.new_zeros(labels.size(0), labels.size(1))
    dec_input[:, 1:] = labels[:, :-1].clone()
    dec_input[:, 0] = tokenizer.bos_token_id                                # <s> w1 w2, ..., wn
    dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)
    return masked_input, attention_mask, dec_input, labels


def get_STD2partial(batch, tokenizer, inp='text', mlm_prob=0.35):
    '''
    If inp == text, then [Masked text -> Text]
    If inp != text, then [Masked Graph -> Graph]
    '''
    assert inp in ["text", "amr"]
    if inp == "text":
        ori_input = batch["input_ids"]
        masked_input = text_infilling(ori_input, tokenizer, mlm_prob=mlm_prob)
        attention_mask = masked_input.ne(tokenizer.pad_token_id).int()
        labels = ori_input.clone()
        labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
        labels = labels[:, 1:]                      # [w1 w2 w3 ...]
        dec_input = ori_input[:, :-1]
        return masked_input, attention_mask, dec_input, labels
    else:
        labels = batch["labels"]                                # [bsz, len+1]
        shifted_input_ids = labels.new_zeros(labels.size(0), labels.size(1) + 1)
        shifted_input_ids[:, 1:] = labels.clone()
        shifted_input_ids[:, 0] = tokenizer.amr_bos_token_id                # <AMR> w1, w2, ..., wn <\s>
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, tokenizer.pad_token_id)   # replace -100 with pad_token_id
        masked_input = text_infilling(shifted_input_ids, tokenizer)
        attention_mask = masked_input.ne(tokenizer.pad_token_id).int()     # attention mask
        dec_input = labels.new_zeros(labels.size(0), labels.size(1))
        dec_input[:, 1:] = labels[:, :-1].clone()
        dec_input[:, 0] = tokenizer.amr_bos_token_id                                # <s> w1 w2, ..., wn
        dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)
        return masked_input, attention_mask, dec_input, labels


def get_MTEG2text(batch, tokenizer, mlm_prob=0.35):
    '''
    [Masked Text + Empty Graph -> text]
    '''
    ori_input = batch["srcEtgt_ids"]
    seg_ids = batch["srcEtgt_segids"]
    masked_input = joint_infilling_partial(ori_input, seg_ids, tokenizer, mask_txt=True, mlm_prob=mlm_prob)
    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()              # attention mask
    labels = batch["input_ids"].clone()     # <s> x1...,xn </s> pad pad
    labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
    labels = labels[:, 1:]                  # x1...,xn </s> pad pad
    dec_input = batch["input_ids"].clone()
    dec_input = dec_input[:, :-1]
    return masked_input, attention_mask, dec_input, labels


def get_ETMG2graph(batch, tokenizer, mlm_prob=0.35):
    '''
    [Empty text + Masked Graph -> graph]
    '''
    ori_input = batch["Esrctgt_ids"]
    seg_ids = batch["Esrctgt_segids"]
    masked_input = joint_infilling_partial(ori_input, seg_ids, tokenizer, mask_txt=False, mlm_prob=mlm_prob)
    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()      # attention mask
    labels = batch["labels"]
    dec_input = labels.new_zeros(labels.size(0), labels.size(1))
    dec_input[:, 1:] = labels[:, :-1].clone()
    dec_input[:, 0] = tokenizer.amr_bos_token_id                        # <s> w1 w2, ..., wn
    dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)
    return masked_input, attention_mask, dec_input, labels


def get_PTPG2partial(batch, tokenizer, inp='text', mlm_prob=0.35):
    '''
    If inp == text, then [Masked text + Graph -> Text]
    If inp != text, then [Text + Masked Graph -> Graph]
    '''
    ori_input = batch["joint_ids"]
    seg_ids = batch["seg_ids"]
    if inp == 'text':
        masked_input = joint_infilling_partial(ori_input, seg_ids, tokenizer, mask_txt=True, mlm_prob=mlm_prob)
        labels = batch["input_ids"].clone()
        labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
        labels = labels[:, 1:]                                                  # w1, w2, .., wn <\s>
        dec_input = batch["input_ids"].clone()
        dec_input = dec_input[:, :-1]                                           # <s> w1, w2, ..., wn
    else:
        masked_input = joint_infilling_partial(ori_input, seg_ids, tokenizer, mask_txt=False, mlm_prob=mlm_prob)
        labels = batch["labels"]
        dec_input = labels.new_zeros(labels.size(0), labels.size(1))
        dec_input[:, 1:] = labels[:, :-1].clone()
        dec_input[:, 0] = tokenizer.amr_bos_token_id                                # <s> w1 w2, ..., wn
        dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)                                  # <AMR> w1 w2, ..., wn

    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()              # attention mask
    return masked_input, attention_mask, dec_input, labels


def get_MTMG2partial(batch, tokenizer, inp='text', mlm_prob=0.35):
    '''
    If inp == text, then [Masked Text + Masked Graph -> text]
    If inp != text, then [Masked Text + Masked Graph -> graph]
    '''
    ori_input = batch["joint_ids"]
    seg_ids = batch["seg_ids"]
    masked_input = joint_infilling_full(ori_input, seg_ids, tokenizer, mlm_prob=mlm_prob)
    if inp == 'text':
        labels = batch["input_ids"].clone()
        labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
        labels = labels[:, 1:]                                                  # w1, w2, .., wn <\s>
        dec_input = batch["input_ids"].clone()
        dec_input = dec_input[:, :-1]                                           # <s> w1 w2, ..., wn
    else:
        labels = batch["labels"]
        dec_input = labels.new_zeros(labels.size(0), labels.size(1))
        dec_input[:, 1:] = labels[:, :-1].clone()
        dec_input[:, 0] = tokenizer.amr_bos_token_id                                # <s> w1 w2, ..., wn
        dec_input.masked_fill_(dec_input == -100, tokenizer.pad_token_id)

    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()              # attention mask
    return masked_input, attention_mask, dec_input, labels


def get_MTMG2TG(batch, tokenizer, mlm_prob=0.35):
    ori_input = batch["joint_ids"]
    seg_ids = batch["seg_ids"]
    masked_input = joint_infilling_full(ori_input, seg_ids, tokenizer, mlm_prob=mlm_prob)

    labels = batch["joint_ids"].clone()
    labels.masked_fill_(labels == tokenizer.pad_token_id, -100)
    labels = labels[:, 1:]                                                  # w1, w2, .., wn <\s>
    dec_input = batch["joint_ids"].clone()
    dec_input = dec_input[:, :-1]                                           # <s> w1 w2, ..., wn
    attention_mask = masked_input.ne(tokenizer.pad_token_id).int()          # attention mask
    return masked_input, attention_mask, dec_input, labels
