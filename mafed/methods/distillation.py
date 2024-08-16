from copy import deepcopy
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler

import wandb
from mafed.data import PrefetchLoader, collate_fn
from mafed.methods import CLStrategy
from mafed.methods.distillation_loss_weights import DistillationWeights


class FeatureDistillation(CLStrategy):
    """Feature Distillation with Separate Vision&Language Weights."""

    def __init__(
        self,
        memory_size,
        opts,
        model_type,
        reg_lambda,
        distillation_modality_weighing_strategy: Literal["equal", "balanced", "adaptive"] = "equal",
        distillation_layer_weighing_strategy: Literal["single", "equal", "discounted"] = "single",
        distillation_coeff=1.0,
        replay_coeff=1.0,
        distillation_layer=-1,
        distillation_loss: Literal["cosine", "mse"] = "mse",
        gamma: float = 0.8,
        num_hidden_layers: int = 11,
        **kwargs,
    ):
        super().__init__(reg_lambda=reg_lambda, opts=opts)
        self.memory_size = memory_size
        num_mem_tasks = len(opts.tasks) - 1
        self.memory_per_task = int(memory_size / num_mem_tasks)
        self.batch_size = opts.batch_size
        self.num_workers = 2
        self.seed = 1

        self.datasets = []
        # random generation for randomly sampling memory data
        self.rng = np.random.default_rng(opts.seed)
        self.pin_mem = opts.pin_mem
        self.step = 0

        self.model_type = model_type
        self.past_model = None
        self.replay_coeff = replay_coeff
        self.distillation_coeff = distillation_coeff
        self.weighing_strategy = distillation_modality_weighing_strategy
        if distillation_loss == "cosine":
            self.distill_loss_fn = torch.nn.CosineEmbeddingLoss(reduction="none")
            self._compute_distillation_loss = self._compute_cosine_distillation_loss
        else:
            self.distill_loss_fn = torch.nn.MSELoss(reduction="none")
            self._compute_distillation_loss = self._compute_mse_distillation_loss
        if distillation_layer is not None and 0 <= distillation_layer < num_hidden_layers:
            hidden_state_layer = distillation_layer
        else:
            hidden_state_layer = None
        self.loss_weights = DistillationWeights(
            distillation_modality_weighing_strategy=distillation_modality_weighing_strategy,
            distillation_layer_weighing_strategy=distillation_layer_weighing_strategy,
            gamma=gamma,
            num_hidden_layers=num_hidden_layers,
            distillation_layer=hidden_state_layer,
        )
        self.opts = opts
        self.num_vision_tokens = 256

    def update(self, dataset, model, dataloader, mask=None, **kwargs):
        self._update_model(model)
        self._update_memory(dataset)
        self.loss_weights.update_weights(model, dataloader, self.task_id)
        self.task_id += 1

    def compute_loss(self, model, loss, batch, **kwargs):
        return loss

    def replay(self, model):
        batch = next(iter(self.mem_dataloader))
        n_ex = batch["input_ids"].size(0)

        do_replay = self.replay_coeff > 0 and self.task_id > 0
        loss = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch, compute_loss=do_replay, output_hidden_states=True, return_dict=True)
            # Replay
            if do_replay:
                loss = self.replay_coeff * output.loss
            if self.distillation_coeff == 0:
                return loss, n_ex
            # Feature Distillation
            dloss = self.distill(output=output, batch=batch)
            if loss is not None:
                loss += dloss
            else:
                loss = dloss
        return loss, n_ex

    def distill(self, output, batch):
        # get hidden statpes from past model
        past_hidden_states = self._get_past_hidden_states(batch)
        total_distill_loss = 0.0
        for layer in self.loss_weights.get_distillation_layers():
            layer_coeff = self.loss_weights.get_layer_loss_weight(layer)
            total_distill_loss += (
                layer_coeff
                * self.distillation_coeff
                * self.feature_distillation(
                    batch=batch,
                    hidden_states=output.hidden_states[layer],
                    past_hidden_states=past_hidden_states[layer],
                    layer=layer,
                )
            )
        self.step += 1
        return total_distill_loss

    def feature_distillation(self, batch, hidden_states, past_hidden_states, layer: int):
        # Distill the [CLS] representation
        if self._cls_distillation:
            cls_loss = self._compute_cls_distillation_loss(
                hidden_states=hidden_states,
                past_hidden_states=past_hidden_states,
            )
            distill_loss = cls_loss
            return distill_loss

        bsz, txt_len = batch["attention_mask"].shape
        language_mask = torch.zeros((bsz, txt_len + self.num_vision_tokens), dtype=batch["attention_mask"].dtype).to(
            batch["attention_mask"].device
        )
        language_mask[:, self.num_vision_tokens :] = batch["attention_mask"]
        batch["lang_masks"] = language_mask
        image_masks = torch.zeros((bsz, txt_len + self.num_vision_tokens), dtype=batch["attention_mask"].dtype).to(
            batch["attention_mask"].device
        )
        image_masks[:, : self.num_vision_tokens] = 1
        batch["image_masks"] = image_masks

        # Distill the language and vision tokens representation
        lang_weight, vision_weight = self.loss_weights.get_modality_loss_weights(
            batch=batch,
            layer=layer,
        )
        # Compute the losses
        lang_loss = self._compute_distillation_loss(
            hidden_states=hidden_states,
            past_hidden_states=past_hidden_states,
            mask=language_mask,
        )

        vision_loss = self._compute_distillation_loss(
            hidden_states=hidden_states,
            past_hidden_states=past_hidden_states,
            mask=image_masks,
        )
        distill_loss = (lang_weight * lang_loss) + (vision_weight * vision_loss)

        wandb.log({f"task_{self.task_id}/distill_loss_{layer}": distill_loss.item()})
        return distill_loss

    def update_after_new_task(self, model, dataset):
        if self.weighing_strategy != "loss_based":
            return
        self.lang_coeff = self.loss_based_distill.update(
            new_model=model, new_dataset=dataset, memory_dataloader=self.mem_dataloader
        )

    def update_after_step(self, model, batch_idx=0, on_train_start=False):
        if self.task_id == 0 or self.weighing_strategy != "dynamic":
            return
        if not self._is_batch_after_step(batch_idx):
            return
        model.vqa_output_distill_loss_params.update()

    def _update_memory(self, dataset):
        if self.task_id > 0:
            del self.mem_dataloader
        # Store a random subset of the task data
        indices = np.arange(len(dataset))
        mem_indices = self.rng.choice(indices, self.memory_per_task, replace=False)
        assert len(set(mem_indices)) == self.memory_per_task
        self.seed = 1
        self.datasets.append(Subset(dataset, mem_indices))

        # Merge with previous samples
        mem_dataset = ConcatDataset(self.datasets)

        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(mem_dataset)
        else:
            sampler = RandomSampler(mem_dataset)

        dataloader = DataLoader(
            mem_dataset,
            sampler=sampler,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_fn["train"][self.model_type],
        )

        self.mem_dataloader = PrefetchLoader(dataloader)
        self.mem_sampler = sampler

    def _update_model(self, model):
        self.past_model = deepcopy(model)
        self.past_model.eval()

    def update_mask(self, mask=None):
        pass

    def _get_past_hidden_states(self, batch):
        with torch.no_grad():
            # get representations from past model
            batch.pop("labels", None)
            hidden_states = self.past_model(**batch, output_hidden_states=True, return_dict=True).hidden_states
            hidden_states = [hh.detach() for hh in hidden_states]
        return hidden_states

    def _compute_cosine_distillation_loss(self, hidden_states, past_hidden_states, mask):
        dim = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(-1, dim)
        past_hidden_states = past_hidden_states.reshape(-1, dim)
        mask = mask.reshape(-1)
        targets = torch.ones_like(mask)
        distill_loss = self.distill_loss_fn(hidden_states, past_hidden_states, targets)
        distill_loss = distill_loss * mask
        distill_loss = distill_loss.sum() / mask.sum()
        return distill_loss

    def _compute_mse_distillation_loss(self, hidden_states, past_hidden_states, mask):
        # Flatten the sequences
        dim = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(-1, dim)
        past_hidden_states = past_hidden_states.reshape(-1, dim)
        mask = mask.reshape(-1)
        # Get the mse per sample
        distill_loss = self.distill_loss_fn(hidden_states, past_hidden_states).sum(-1) / dim
        # Zero-out the mse of masked tokens
        distill_loss = distill_loss * mask
        # Average by the number of non-zero tokens
        distill_loss = distill_loss.sum() / mask.sum()
        return distill_loss

    def _compute_cls_distillation_loss(self, hidden_states, past_hidden_states):
        hidden_states = hidden_states[:, 0]
        past_hidden_states = past_hidden_states[:, 0]
        targets = torch.ones(hidden_states.shape[0]).to(hidden_states.device)
        distill_loss = self.distill_loss_fn(hidden_states, past_hidden_states, targets)

        return distill_loss.mean()
