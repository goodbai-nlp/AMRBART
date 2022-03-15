# coding:utf-8
import os
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from pytorch_lightning.utilities import rank_zero_info
from common.constant import (
    special_tokens,
    arg_to_scheduler,
    ROUGE_KEYS,
    arg_to_scheduler_choices,
    arg_to_scheduler_metavar,
)
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    Adafactor,
    AutoConfig,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)
import pytorch_lightning as pl
from common.utils import (
    save_dummy_batch,
    calculate_bleu,
    flatten_list,
    label_smoothed_nll_loss,
    lmap,
    save_json,
    convert_text,
    eval_bleu_sents,
    eval_bleu_sents_tok,
)


class AMR2TextModelModule(pl.LightningModule):
    loss_names = ["loss"]
    metric_names = ["bleu"]
    default_val_metric = "bleu"

    def __init__(self, tokenizer, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.metrics = defaultdict(list)
        if args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            raise ValueError(
                "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --config_name"
            )
        self.modify_config(args, config, ("dropout", "attention_dropout"))
        self.tokenizer = tokenizer
        
        if args.model_name_or_path:
            print(f"Loading pretrained model from {args.model_name_or_path}")
            self.model = BartForConditionalGeneration.from_pretrained(
                args.model_name_or_path, cache_dir=args.cache_dir, from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=config,
            )
        else:
            self.logger.info("Model name or path is not provided")
            exit()
        
        print("Ori EMbeddings: ", self.model.model.shared.num_embeddings)
        self.model.resize_token_embeddings(len(self.tokenizer))
        print("Resized EMbeddings: ", self.model.model.shared.num_embeddings)
        if args.smart_init:
            self.model = self.smart_emb_init_new()

        self.val_metric = (
            self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric
        )
        self.step_count = 0
        self.val_count = -1
        self.saved_dummy = False
        self.vocab_size = len(self.tokenizer)
        self.eval_beams = self.hparams.eval_beam
        self.eval_lenpen = self.hparams.eval_lenpen
        self.eval_max_length = self.hparams.eval_max_length
        self.decoder_start_token_id = self.tokenizer.bos_token_id
        self.decoder_end_token_id = self.tokenizer.eos_token_id

    def setup(self, stage=None):
        if stage == "fit":
            num_devices = max(1, self.hparams.gpus)
            effective_batch_size = (
                self.hparams.per_gpu_train_batch_size
                * self.hparams.accumulate_grad_batches
                * num_devices
            )
            print(f"Effective batch size: {effective_batch_size}")
            if self.hparams.max_steps <= 0:
                self.total_steps = (
                    self.hparams.train_dataset_size / effective_batch_size
                ) * self.hparams.max_epochs
                print(f"Effective training steps: {self.total_steps}")
            else:
                self.total_steps = self.hparams.max_steps
                effective_epochs = self.hparams.max_steps / (
                    self.hparams.train_dataset_size / effective_batch_size
                )
                print(f"Effective training epoches: {effective_epochs}")

    def smart_emb_init_new(self):
        print("Initializing AMR Vocab according to similar tokens ...")
        for tok, idx in self.tokenizer.encoder.items():
            tok = tok.lstrip(self.tokenizer.INIT)

            if idx < self.tokenizer.old_enc_size:
                continue

            elif tok.startswith('<pointer:') and tok.endswith('>'):
                tok_split = ['pointer', str(tok.split(':')[1].strip('>'))]

            elif tok.startswith('<'):
                continue

            elif tok.startswith(':'):

                if tok.startswith(':op'):
                    tok_split = ['relation', 'operator', str(int(tok[3:]))]

                elif tok.startswith(':snt'):
                    tok_split = ['relation', 'sentence', str(int(tok[4:]))]

                elif tok.startswith(':ARG'):
                    tok_split = ['relation', 'argument', str(int(tok[4:]))]

                else:
                    tok_split = ['relation'] + tok.lstrip(':').split('-')

            else:
                tok_split = tok.split('-')

            tok_split_ = tok_split
            tok_split = []
            for s in tok_split_:
                s_ = s + self.tokenizer.INIT
                if s_ in self.tokenizer.encoder:
                    # print(f"{s_} in tokenizer vocabulary")
                    tok_split.append(s_)
                else:
                    tok_split.extend(self.tokenizer._tok_bpe(s))     # 
            
            vecs = []
            for s in tok_split:
                idx_split = self.tokenizer.encoder.get(s, -1)
                if idx_split > -1:
                    vec_split = self.model.model.shared.weight.data[idx_split].clone()
                    vecs.append(vec_split)

            if vecs:
                vec = torch.stack(vecs, 0).mean(0)
                noise = torch.empty_like(vec)
                noise.uniform_(-0.1, +0.1)
                self.model.model.shared.weight.data[idx] = vec + noise
        
        return self.model

    def modify_config(self, args, config, modified_params=("dropout")):
        for p in modified_params:
            if getattr(args, p, None):
                assert hasattr(config, p), f"model config doesn't have a `{p}` attribute"
                setattr(config, p, getattr(args, p))
                print("Manually set:", p, getattr(args, p))
            else:
                print("Args don't have:", p)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": self.hparams.learning_rate,
            }
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        scheduler = self.get_lr_scheduler(optimizer)
        return [optimizer], [scheduler]

    def get_lr_scheduler(self, optimizer):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == "constant":
            scheduler = get_schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        elif self.hparams.lr_scheduler == "cosine_w_restarts":
            scheduler = get_schedule_func(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.total_steps,
                num_cycles=self.hparams.max_epochs,
            )
        else:
            scheduler = get_schedule_func(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.total_steps,
            )
        # print("scheduler total steps:", self.total_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = Path(self.hparams.output_dir).joinpath("best_tfmr")
        self.model.config.save_step = self.val_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def _step(self, batch: dict, eval=False) -> Tuple:
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        decoder_input_ids, tgt_ids = batch["decoder_input_ids"], batch["labels"]

        if not self.saved_dummy:
            save_dummy_batch(batch, self.tokenizer, self.hparams.output_dir)
            self.saved_dummy = True

        outputs = self.model(
            input_ids=src_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=False,
        )
        lm_logits = outputs[0]
        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.label_pad)
            assert lm_logits.shape[-1] == self.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=self.label_pad
            )
        return (loss,)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def label_pad(self) -> int:
        # return -100
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch, eval=False)
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.label_pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
        logs["lr"] = self.trainer.lr_schedulers[0]["scheduler"].get_lr()[0]
        # self.log("train_loss", loss_tensors[0].detach(), prog_bar=True)
        self.log("lr", self.trainer.lr_schedulers[0]["scheduler"].get_lr()[0], prog_bar=True)
        return {"loss": loss_tensors[0], "logs": logs}

    def training_epoch_end(self, outputs, prefix="train") -> Dict:
        losses = {k: torch.stack([x[k] for x in outputs]).mean().item() for k in self.loss_names}
        self.metrics["training"].append(losses)

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.val_count += 1
        print(f"Generating Kwargs: Num_beam: {self.eval_beams}, Max_len: {self.eval_max_length}")
        # print('ori outputs', outputs)
        outputs = self.all_gather(outputs)
        # print('Gathered outputs', outputs)
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        generative_metrics = {
            k: torch.stack([x[k] for x in outputs]).mean().item()
            for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metric_val = (
            generative_metrics[self.val_metric]
            if self.val_metric in generative_metrics
            else losses[self.val_metric]
        )
        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["val_count"] = self.val_count
        # print('all_metrics:', all_metrics)
        self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])

        val_outputs_folder = "val_outputs"
        os.system("mkdir -p " + os.path.join(self.hparams.output_dir, val_outputs_folder))

        if "preds" in outputs[0]:
            tb_all = {}
            idx_tb = 0
            for output_batch in outputs:
                a, b, c, e = (
                    output_batch["a"],
                    output_batch["b"],
                    output_batch["c"],
                    output_batch["e"],
                )

                for aa, bb, ee, cc in zip(a, b, e, c):
                    tb_all[idx_tb] = {}
                    tb_all[idx_tb]["input_ids"] = aa
                    tb_all[idx_tb]["labels"] = bb
                    tb_all[idx_tb]["decoder_input_ids"] = ee
                    tb_all[idx_tb]["generated_ids"] = cc
                    idx_tb += 1

            file_debug = os.path.join(
                self.hparams.output_dir,
                val_outputs_folder,
                "debug_" + str(self.val_count) + ".json",
            )
            save_json(tb_all, file_debug)

            output_test_predictions_file = os.path.join(
                self.hparams.output_dir,
                val_outputs_folder,
                "validation_predictions_" + str(self.val_count) + ".txt",
            )
            output_test_targets_file = os.path.join(
                self.hparams.output_dir,
                val_outputs_folder,
                "validation_targets_" + str(self.val_count) + ".txt",
            )
            # write predictions and targets for later rouge evaluation.
            # print("targets:", output_batch["target"])
            with open(output_test_predictions_file, "w") as p_writer, open(output_test_targets_file, "w") as t_writer:
                for output_batch in outputs:
                    p_writer.writelines(s + "\n" for s in output_batch["preds"])
                    t_writer.writelines(s + "\n" for s in output_batch["target"])
                p_writer.close()
                t_writer.close()

            bleu_info = eval_bleu_sents(output_test_targets_file, output_test_predictions_file)
            bleu_info_tok = eval_bleu_sents_tok(
                output_test_targets_file, output_test_predictions_file
            )
            # chrf_info = eval_chrf(output_test_targets_file, output_test_predictions_file)

            rank_zero_info("number epoch: %s", self.val_count)
            rank_zero_info("%s corpus_bleu_info: %s", self.val_count, bleu_info)
            rank_zero_info("%s corpus_bleu_info_tok: %s", self.val_count, bleu_info_tok)
            # rank_zero_info("%s chrf_info: %s", self.val_count, chrf_info)

            # exit()
        self.log_dict(all_metrics, sync_dist=True)
        return {
            "log": all_metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

    def calc_generative_metrics(self, preds, target) -> Dict:
        # return calculate_rouge(preds, target)
        return calculate_bleu(preds, target)
    
    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        decoder_input_ids, lm_labels = batch["decoder_input_ids"], batch["labels"]

        generated_ids = self.model.generate(
            src_ids,
            attention_mask=src_mask,
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            eos_token_id=self.decoder_end_token_id,
            num_beams=self.eval_beams,
            no_repeat_ngram_size=0,
            min_length=0,
            max_length=self.eval_max_length,
            length_penalty=self.eval_lenpen,
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)                    #
        target: List[str] = self.ids_to_clean_text(batch["labels"])                 #
        # preds: List[str] = self.tokenizer.batch_decode(generated_ids.tolist())
        # target: List[str] = self.tokenizer.batch_decode(batch["labels"].tolist())     #

        a = self.tokenizer.batch_decode(batch["input_ids"].tolist())    # gold input
        b = self.tokenizer.batch_decode(lm_labels.tolist())             # gold predict
        c = self.tokenizer.batch_decode(generated_ids)                  # generated
        e = self.tokenizer.batch_decode(decoder_input_ids.tolist())     # decoder input

        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        score: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(
            gen_time=gen_time,
            gen_len=summ_len,
            preds=preds,
            target=target,
            a=a,
            b=b,
            c=c,
            e=e,
            **score,
        )
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")
