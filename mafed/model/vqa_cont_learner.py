"""
Continual Learning for VQA
"""

import os
from math import ceil

import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam, Adamax

from mafed.model import model_architecture
from mafed.optim.adamw import AdamW
from mafed.optim.sched import get_linear_schedule_with_warmup
from mafed.utils.eval_utils import VQAGenerativeAccuracy
from mafed.utils.logger import LOGGER


class BaseModule(LightningModule):
    def initialization_state_dict(self, checkpoint):
        """Get the state dict to initialize the model."""
        state_dict = {}
        if checkpoint is not None:
            LOGGER.info(f"Loading {checkpoint}")
            state_dict = torch.load(checkpoint)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            state_dict = {self._prep_module_name(module): params for module, params in state_dict.items()}
        return state_dict

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:  # noqa: WPS337
            dataset_size = self.trainer.limit_train_batches
            num_devices = 1
        else:
            if isinstance(self.trainer.limit_train_batches, float) and self.trainer.limit_train_batches != 1.0:
                # limit_train_batches is a percentage of batches
                dataset_size = len(
                    # type: ignore[attr-defined]
                    self.trainer.datamodule.train_dataloader()
                )
                dataset_size = int(dataset_size * self.trainer.limit_train_batches)
            else:
                dataset_size = len(self.trainer.datamodule.train_dataloader())  # type: ignore[unreachable]
            num_devices = max(1, self.trainer.num_devices)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices  # type: ignore[attr-defined]
        num_steps = ceil(dataset_size / effective_batch_size) * self.trainer.max_epochs
        # num_steps = max(num_steps, dataset_size)
        if self.trainer.max_steps and 0 < self.trainer.max_steps < num_steps:
            num_steps = self.trainer.max_steps

        LOGGER.info(f"Total number of training steps: {num_steps}")

        return num_steps

    def compute_warmup(self) -> tuple:
        """Compute the number of total training and warmup steps."""
        total_steps = self.num_training_steps()

        dataset_size = len(self.trainer.datamodule.train_dataloader())
        total_steps = ceil(dataset_size / self.trainer.accumulate_grad_batches) * 60
        if "warmup_steps" in vars(self.config):
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = int(self.config.warmup_perc * total_steps)

        return total_steps, int(warmup_steps)

    def configure_optimizers(self) -> dict:
        """Configure the optimizer and learning rate scheduler."""
        no_decay = [
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
            "vqa_output_distill_loss_params",
        ]
        param_optimizer = [(n, p) for n, p in self.model.named_parameters() if not self._in_top_param(n)]
        param_top = [(n, p) for n, p in self.model.named_parameters() if self._in_top_param(n)]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_top if not any(nd in n for nd in no_decay)],
                "lr": self.config.lr_mul * self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in param_top if any(nd in n for nd in no_decay)],
                "lr": self.config.lr_mul * self.config.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "lr": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "lr": self.config.learning_rate,
                "weight_decay": 0.0,
            },
        ]

        # currently Adam only
        if self.config.optim == "adam":
            OptimCls = Adam
        elif self.config.optim == "adamax":
            OptimCls = Adamax
        elif self.config.optim == "adamw":
            OptimCls = AdamW
        else:
            raise ValueError("invalid optimizer")
        optimizer = OptimCls(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )

        total_steps, warmup_steps = self.compute_warmup()
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps, last_epoch=-1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _in_top_param(self, param_name):
        include_distill_param_in_top = True
        cond = "vqa_output" in param_name
        if not include_distill_param_in_top:
            cond = cond and "distill_loss" not in param_name
        return cond

    def _prep_module_name(self, module: str) -> str:
        if module.startswith("model."):
            return module[len("model.") :]
        return module


class VLPythiaVQAModule(BaseModule):
    """VQA Module"""

    def __init__(self, opts, tokenizer):
        super().__init__()
        self.config = opts
        self.model = model_architecture["vlpythia"].from_pretrained(
            opts.model_name,
            vision_encoder_name=opts.vision_encoder_name,
            use_flash_attention_2=True,
        )

        LOGGER.info("BRRR The vision encoder is freezing")
        for p in self.model.vision_encoder.parameters():
            p.requires_grad = False
        self._tokenizer = tokenizer
        self.vqa_accuracy = VQAGenerativeAccuracy()

    def training_step(self, batch, batch_idx):
        """Forward pass."""
        loss = self.model(**batch, compute_loss=True, return_dict=True).loss
        self.log("train_loss", loss, batch_size=batch["labels"].shape[0], on_step=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        LOGGER.info(f"Validation accuracy = {self.vqa_accuracy.compute(): .4f}")
        self.vqa_accuracy.reset()
        return super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            max_new_tokens=10,
            use_cache=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        predictions = self._tokenizer.batch_decode(outputs[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
        # for output in outputs:
        #     predictions.append(self._tokenizer.decode(output[batch["input_ids"].shape[0]:], skip_special_tokens=True))
        self.vqa_accuracy(predictions, batch["answers"])
        self.log("valid_accuracy", self.vqa_accuracy, on_step=False, on_epoch=True)
        return predictions


class VLPythiaVQACLearner(BaseModule):
    """VQA Continual Learner."""

    def __init__(self, task_id, opts, tokenizer, cl_method):
        super().__init__()
        self.task_id = task_id
        self.config = opts
        self.model = model_architecture["vlpythia"].from_pretrained(
            opts.model_name,
            vision_encoder_name=opts.vision_encoder_name,
            use_flash_attention_2=True,
        )
        LOGGER.info("BRRR The vision encoder is freezing")
        for p in self.model.vision_encoder.parameters():
            p.requires_grad = False
        self._tokenizer = tokenizer
        self.vqa_accuracy = VQAGenerativeAccuracy()

        self.cl_method = cl_method

    def on_train_start(self):
        """After initializing for the new task, before training."""
        self.cl_method.num_training_steps = self.num_training_steps()

    def training_step(self, batch, batch_idx):
        """Forward pass."""
        loss = None
        if self.task_id > 0 and (batch_idx + 1) % self.config.replay_interval == 0:
            loss, _ = self.cl_method.replay(self.model)
        if loss is None:
            loss = self.model(**batch, compute_loss=True, return_dict=True).loss
            loss = self.cl_method.compute_loss(self.model, loss, batch=batch)
            self.log(
                f"task_{self.task_id}/train_loss",
                loss,
                batch_size=batch["input_ids"].shape[0],
                on_step=True,
                prog_bar=True,
            )
        else:
            self.log(
                f"task_{self.task_id}/replay_train_loss",
                loss,
                batch_size=batch["input_ids"].shape[0],
                on_step=True,
                prog_bar=True,
            )
        return loss

    def on_before_optimizer_step(self, optimizer):
        """Update after backward to have access to model gradients."""
        distill_loss_param = getattr(self.model, "vqa_output_distill_loss_params", None)
        if distill_loss_param is not None and distill_loss_param.model.params.grad is not None:
            alpha_param = distill_loss_param.model.params.clone().detach()
            self.log(f"task_{self.task_id}/alpha_value", torch.sigmoid(alpha_param).item())
            self.log(
                f"task_{self.task_id}/alpha_grad",
                distill_loss_param.model.params.grad.item(),
            )

        self.cl_method.update_after_backward(model=self.model)
        super().on_before_optimizer_step(optimizer)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update after the optimization step, i.e. with the most recent parameters."""
        self.cl_method.update_after_step(model=self.model, batch_idx=batch_idx)

    def on_validation_epoch_end(self) -> None:
        LOGGER.info(f"Validation accuracy = {self.vqa_accuracy.compute(): .4f}")
        return super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            max_new_tokens=10,
            use_cache=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        predictions = self._tokenizer.batch_decode(outputs[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
        self.vqa_accuracy(predictions, batch["answers"])
        self.log(
            f"task_{self.task_id}/valid_acc",
            self.vqa_accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=batch["input_ids"].shape[0],
        )
