# coding:utf-8
import os
import json

from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    Adafactor,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartTokenizer,
    BartForConditionalGeneration,
    T5Tokenizer,
    T5Model,
    T5ForConditionalGeneration,
)

raw_special_tokens = json.load(
    open(f"{os.path.dirname(__file__)}/additional-tokens.json", "r", encoding="utf-8")
)
special_tokens = [itm.lstrip("Ġ") for itm in raw_special_tokens]

recategorizations = [
    "\u0120COUNTRY",
    "\u0120QUANTITY",
    "\u0120ORGANIZATION",
    "\u0120DATE_ATTRS",
    "\u0120NATIONALITY",
    "\u0120LOCATION",
    "\u0120ENTITY",
    "\u0120MISC",
    "\u0120ORDINAL_ENTITY",
    "\u0120IDEOLOGY",
    "\u0120RELIGION",
    "\u0120STATE_OR_PROVINCE",
    "\u0120CAUSE_OF_DEATH",
    "\u0120TITLE",
    "\u0120DATE",
    "\u0120NUMBER",
    "\u0120HANDLE",
    "\u0120SCORE_ENTITY",
    "\u0120DURATION",
    "\u0120ORDINAL",
    "\u0120MONEY",
    "\u0120CRIMINAL_CHARGE",
]

# special_tokens = ["<AMR>", "</AMR>"]

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

arg_to_tokenizer = {
    "AutoTokenizer": AutoTokenizer,
    "BartTokenizer": BartTokenizer,
    "T5Tokenizer": T5Tokenizer,
}
arg_to_plm_model = {
    "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
    "BartForConditionalGeneration": BartForConditionalGeneration,
    "T5Model": T5Model,
    "T5ForConditionalGeneration": T5ForConditionalGeneration,
}
