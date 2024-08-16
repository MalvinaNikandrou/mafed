import json
import pickle
from os.path import join
from random import shuffle

import torch.distributed as dist


def random_task_order(exp_name, root_task_ids):
    split_file = join(root_task_ids, exp_name, f"train_question_ids.json")
    with open(split_file, "r") as fp:
        tasks = list(json.load(fp).keys())
    shuffle(tasks)
    return tasks
