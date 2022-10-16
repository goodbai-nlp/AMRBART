# coding:utf-8

import logging
import os
import torch
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
from .utils import save_json
logger = logging.getLogger(__name__)


def get_early_stopping_callback(metric, patience):
    return EarlyStopping(
        monitor=f"val_avg_{metric}",                    # does this need avg?
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True,
    )


def get_checkpoint_callback(output_dir, metric, save_top_k=1, lower_is_better=False, save_interval=-1):
    """Saves the best model by validation {metric} score."""
    if metric == "rouge2":
        exp = "{val_avg_rouge2:.4f}-{val_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.4f}-{val_count}"
    elif metric == "loss":
        exp = "{val_avg_loss:.4f}-{val_count}"
    elif metric == "smatch":
        exp = "{val_avg_smatch:.4f}-{val_count}"
    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support rouge2, bleu and loss, got {metric}, You can make your own by adding to this function."
        )
    # print("valid_metric", exp)
    if save_interval < 0:
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename=exp,
            monitor=f"val_avg_{metric}",
            mode="min" if "loss" in metric else "max",
            save_top_k=save_top_k,
            every_n_epochs=1,
            save_last=True,
        )
    else:
        if metric == "rouge2":
            exp = "{val_avg_rouge2:.4f}-{step}"
        elif metric == "bleu":
            exp = "{val_avg_bleu:.4f}-{step}"
        elif metric == "loss":
            exp = "{val_avg_loss:.4f}-{step}"
        elif metric == "smatch":
            exp = "{val_avg_smatch:.4f}-{step}"
        else:
            raise NotImplementedError(
                f"seq2seq callbacks only support rouge2, bleu and loss, got {metric}, You can make your own by adding to this function."
            )
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename=exp,
            monitor=f"val_avg_{metric}",
            mode="min" if "loss" in metric else "max",
            save_top_k=save_top_k,
            every_n_train_steps=save_interval,
            save_last=True,
        )

    return checkpoint_callback


class LoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_lr())}
        pl_module.logger.log_metrics(lrs)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Training results *****")
        metrics = trainer.callback_metrics
        rank_zero_info(trainer.logger)
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        rank_zero_info(trainer.logger)
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class Seq2SeqLoggingCallback(pl.Callback):
    def on_batch_end(self, trainer, pl_module):
        lrs = {
            f"lr_group_{i}": param["lr"]
            for i, param in enumerate(pl_module.trainer.optimizers[0].param_groups)
        }
        pl_module.logger.log_metrics(lrs)

    @rank_zero_only
    def _write_logs(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        type_path: str,
        save_generations=True,
    ) -> None:
        logger.info(f"***** {type_path} results at step {trainer.global_step:05d} *****")
        metrics = trainer.callback_metrics
        trainer.logger.log_metrics(
            {k: v for k, v in metrics.items() if k not in ["log", "progress_bar", "preds"]}
        )
        # Log results
        od = Path(pl_module.hparams.output_dir)
        if type_path == "test":
            results_file = od / "test_results.txt"
            generations_file = od / "test_generations.txt"
        else:
            # this never gets hit. I prefer not to save intermediate generations, and results are in metrics.json
            # If people want this it will be easy enough to add back.
            results_file = od / f"{type_path}_results/{trainer.global_step:05d}.txt"
            generations_file = od / f"{type_path}_generations/{trainer.global_step:05d}.txt"
            results_file.parent.mkdir(exist_ok=True)
            generations_file.parent.mkdir(exist_ok=True)
        with open(results_file, "a+") as writer:
            for key in sorted(metrics):
                if key in ["log", "progress_bar", "preds"]:
                    continue
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                msg = f"{key}: {val:.6f}\n"
                writer.write(msg)

        if not save_generations:
            return

        if "preds" in metrics:
            content = "\n".join(metrics["preds"])
            generations_file.open("w+").write(content)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        try:
            npars = pl_module.model.model.num_parameters()
        except AttributeError:
            npars = pl_module.model.num_parameters()

        n_trainable_pars = count_trainable_parameters(pl_module)
        # mp stands for million parameters
        trainer.logger.log_metrics(
            {"n_params": npars, "mp": npars / 1e6, "grad_mp": n_trainable_pars / 1e6}
        )

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        save_json(pl_module.metrics, pl_module.metrics_save_path)
        return self._write_logs(trainer, pl_module, "test")

    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        save_json(pl_module.metrics, pl_module.metrics_save_path)

        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar", "preds"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
        # Uncommenting this will save val generations
        # return self._write_logs(trainer, pl_module, "valid")
