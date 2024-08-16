"""
Dataset interfaces
"""

import itertools
import json
import os

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        data_path,
        split_file="splits.json",
        task=None,
        split=None,
    ):
        """
        data_path (str, optional): Path to VQA annotations.
        split_file (str, optional): Path to file containing the question ids for each task and split.
        task (str, optinal): Use only the samples that belong to this task.
        split (str, optional): [train, val] Use only the samples that belong to this split. 
        """
        ids = self.prepare_ids(split_file, task)
        with open(os.path.join(data_path, f"{split}_annotations.json"), "r") as f:
            qid_to_annotations = json.load(f)
        self.annotations = [qid_to_annotations[qid] for qid in ids]

    def prepare_ids(self, split_file="splits.json", task=None):
        if task and split_file:
            assert os.path.exists(split_file)
            if os.path.exists(split_file):
                with open(split_file, "r") as fp:
                    splits_ids = json.load(fp)
                # Joint task refers to multitask learning
                if task == "joint":
                    ids = list(itertools.chain.from_iterable([splits_ids[t] for t in splits_ids]))
                elif task in splits_ids:
                    ids = splits_ids[task]
                else:
                    raise ValueError(f"Invalid task: {task}")
            else:
                raise ValueError(f"Incorrect splits file {split_file}")
        else:
            raise ValueError(f"No question ids for task: {task} and task ids file: {split_file}")
        return ids

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, i):
        example = self.annotations[i]
        return example
