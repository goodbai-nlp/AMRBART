# coding:utf-8
import torch
import itertools
import json
import random
import numpy as np
import nltk
import os
import re
import smatch
from pathlib import Path
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from typing import Dict, List, Tuple
from transformers import PreTrainedTokenizer, EvalPrediction
from typing import Callable, Dict, Iterable, List, Tuple, Union
from collections import Counter
from common.constant import ROUGE_KEYS



def add_newline_to_end_of_each_sentence(x: str) -> str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    assert nltk, "nltk must be installed to separate newlines between sentences. (pip install nltk)"
    return "\n".join(nltk.sent_tokenize(x))


def set_seed(seed):
    # print(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mask_tokens(
    inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(labels.device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8).to(labels.device)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5).to(labels.device)).bool() & masked_indices & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def save_dummy_batch(batch, tokenizer, output_dir):
    print("Saving dummy inputs...")
    json_out_path = open(output_dir + "/dummy_input.json", "w", encoding="utf-8")
    ith_dict = {}
    # print('Input Id Size:', batch["input_ids"].size())
    ith_dict["input_ids"] = str(batch["input_ids"].tolist())
    ith_dict["input_tokens"] = tokenizer.batch_decode(batch["input_ids"].tolist())
    label_data = batch["labels"].tolist()
    ith_dict["label_ids"] = str(label_data)
    # label_data_new = [[idx if idx!=-100 else tokenizer.pad_token_id for idx in ith_label] for ith_label in label_data]
    ith_dict["label_tokens"] = tokenizer.batch_decode(label_data)
    ith_dict["dec_input_ids"] = str(batch["decoder_input_ids"].tolist())
    ith_dict["dec_input_tokens"] = tokenizer.batch_decode(batch["decoder_input_ids"].tolist())
    # ith_dict["amr_ids"] = str(batch["amr_ids"].tolist())
    # ith_dict["amr_tokens"] = tokenizer.batch_decode(batch["amr_ids"].tolist())
    json.dump(ith_dict, json_out_path, indent=4)


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


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


def convert_text(text):
    # return text
    text = " ".join(re.split("(\W)", text))
    text = " ".join(text.split())
    return text


def eval_bleu_sents(ref_file, pred_file):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../../utils"
    cmd_string = (
        "perl "
        + folder_data_before
        + "/multi-bleu.perl -lc "
        + ref_file
        + " < "
        + pred_file
        + " > "
        + pred_file.replace("txt", "bleu")
    )

    os.system(cmd_string)

    try:
        bleu_info = open(pred_file.replace("txt", "bleu"), "r").readlines()[0].strip()
    except:
        bleu_info = "no data"

    return bleu_info


def eval_bleu_sents_tok(ref_file, pred_file):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../../utils"

    cmd_string = (
        "perl "
        + folder_data_before
        + "/tokenizer.perl -threads 4 -no-escape < "
        + pred_file
        + " > "
        + pred_file
        + "_tok"
    )
    os.system(cmd_string)
    cmd_string = (
        "perl "
        + folder_data_before
        + "/tokenizer.perl -threads 4 -no-escape < "
        + ref_file
        + " > "
        + ref_file
        + "_tok"
    )
    os.system(cmd_string)
    cmd_string = (
        "perl "
        + folder_data_before
        + "/multi-bleu.perl -lc "
        + ref_file
        + "_tok"
        + " < "
        + pred_file
        + "_tok"
        + " > "
        + pred_file.replace("txt", "bleu_tok")
    )
    os.system(cmd_string)

    try:
        bleu_info_data = open(pred_file.replace("txt", "bleu_tok"), "r").readlines()[0].strip()
    except:
        bleu_info_data = "no data"

    return bleu_info_data


def eval_meteor(ref_file, pred_file):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../../utils"

    cmd_string = (
        "java -jar "
        + folder_data_before
        + "/meteor-1.5.jar "
        + pred_file
        + " "
        + ref_file
        + " > "
        + pred_file.replace("txt", "meteor")
    )

    os.system(cmd_string)

    meteor_info = open(pred_file.replace("txt", "meteor"), "r").readlines()[-1].strip()

    return meteor_info


def eval_chrf(ref_file, pred_file):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../../utils"

    cmd_string = (
        "python "
        + folder_data_before
        + "/chrf++.py -H "
        + pred_file
        + " -R "
        + ref_file
        + " > "
        + pred_file.replace("txt", "chrf")
    )

    os.system(cmd_string)

    try:
        chrf_info_1 = open(pred_file.replace("txt", "chrf"), "r").readlines()[1].strip()
        chrf_info_2 = open(pred_file.replace("txt", "chrf"), "r").readlines()[2].strip()
        chrf_data = chrf_info_1 + " " + chrf_info_2
    except:
        chrf_data = "no data"

    return chrf_data


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    # nll_loss = nll_loss.sum()                   # mean()? Scared to break other math.
    # smooth_loss = smooth_loss.sum()
    nll_loss = nll_loss.mean()                   # mean()? Scared to break other math.
    smooth_loss = smooth_loss.mean()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": round(corpus_bleu(output_lns, [refs_lns], **kwargs).score, 4)}


def build_compute_metrics_fn(
    task_name: str, tokenizer: PreTrainedTokenizer
) -> Callable[[EvalPrediction], Dict]:
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        return pred_str, label_str

    def summarization_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        rouge.update({"gen_len": summ_len})
        return rouge

    def translation_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        bleu: Dict = calculate_bleu(pred_str, label_str)
        gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        bleu.update({"gen_len": gen_len})
        return bleu

    compute_metrics_fn = (
        summarization_metrics if "summarization" in task_name else translation_metrics
    )
    return compute_metrics_fn


def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {
            stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]
        }
    return new_dict


def calculate_rouge(
    pred_lns: List[str],
    tgt_lns: List[str],
    use_stemmer=True,
    rouge_keys=ROUGE_KEYS,
    return_precision_and_recall=False,
    bootstrap_aggregation=True,
    newline_sep=True,
) -> Dict:
    """Calculate rouge using rouge_scorer package.

    Args:
        pred_lns: list of summaries generated by model
        tgt_lns: list of groundtruth summaries (e.g. contents of val.target)
        use_stemmer:  Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching.
        rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
        return_precision_and_recall: (False) whether to also return precision and recall.
        bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
            this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
        newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
        on multi sentence summaries (CNN/DM dataset).

    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys

    """

    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(tgt_lns, pred_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            tgt = add_newline_to_end_of_each_sentence(tgt)
            # pred = pred + '\n'
            # tgt = tgt + '\n'
        scores = scorer.score(pred, tgt)
        aggregator.add_scores(scores)

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)


def calculate_smatch(test_path, predictions_path) -> dict:
    with Path(predictions_path).open() as p, Path(test_path).open() as g:
        score = next(smatch.score_amr_pairs(p, g))
    return {"smatch": score[2]}