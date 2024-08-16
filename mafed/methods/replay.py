"""
Implementation of CL Replay methods:
ER (Experience Replay).
"""

import numpy as np
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from mafed.data import PrefetchLoader, collate_fn
from mafed.methods import CLStrategy


class ER(CLStrategy):
    def __init__(self, opts, memory_size, model_type, **kwargs):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        num_mem_tasks = len(opts.tasks) - 1
        self.memory_per_task = int(memory_size / num_mem_tasks)

        self.datasets = []
        # random generation for randomly sampling memory data
        self.rng = np.random.default_rng(opts.seed)
        self.num_workers = opts.n_workers
        self.pin_mem = opts.pin_mem
        self.batch_size = opts.batch_size
        self.seed = 1

        self.model_type = model_type

    def update(self, dataset, **kwargs):
        if self.task_id > 0:
            del self.mem_dataloader
        self.task_id += 1
        # Store a random subset of the task data
        indices = np.arange(len(dataset))
        mem_indices = self.rng.choice(indices, self.memory_per_task, replace=False)

        self.seed = 1
        self.datasets.append(Subset(dataset, mem_indices))

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

    def update_mask(self, mask=None):
        self.mask = mask

    def compute_loss(self, model, loss, **kwargs):
        return loss

    def replay(self, model):
        batch = next(iter(self.mem_dataloader))
        n_ex = batch["input_ids"].size(0)
        loss = model(**batch, compute_loss=True, return_dict=True).loss
        return loss, n_ex
