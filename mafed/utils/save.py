"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

saving utilities
"""

import json
import os
import subprocess
from os.path import abspath, dirname, join

import torch.distributed as dist

from mafed.utils.logger import LOGGER, add_log_to_file


def save_configs(opts, ans2label=None):
    """Save configurations of the experiment."""
    save_training_meta(opts)
    # save task order for validation
    tasks_file = join(opts.output_dir, "ckpt", "task_order.json")
    with open(tasks_file, "w") as out_file:
        json.dump(opts.tasks, out_file)
    add_log_to_file(join(opts.output_dir, "log", "log.txt"))
    # Save ans2label
    if ans2label is None:
        return
    save_ans2label = join(opts.output_dir, "ckpt", "ans2label.json")
    with open(save_ans2label, "w") as out_file:
        json.dump(ans2label, out_file)


def save_training_meta(args):
    if dist.is_available() and dist.is_initialized() and dist.get_rank() > 0:
        return

    os.makedirs(join(args.output_dir, "log"), exist_ok=True)
    os.makedirs(join(args.output_dir, "ckpt"), exist_ok=True)

    with open(join(args.output_dir, "log", "hps.json"), "w") as writer:
        json.dump(vars(args), writer, indent=4)
    # model_config = json.load(open(args.model_config))
    # with open(join(args.output_dir, "log", "model.json"), "w") as writer:
    #     json.dump(model_config, writer, indent=4)
    # git info
    try:
        LOGGER.info("Waiting on git info....")
        c = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            timeout=10,
            stdout=subprocess.PIPE,
        )
        git_branch_name = c.stdout.decode().strip()
        LOGGER.info("Git branch: %s", git_branch_name)
        c = subprocess.run(["git", "rev-parse", "HEAD"], timeout=10, stdout=subprocess.PIPE)
        git_sha = c.stdout.decode().strip()
        LOGGER.info("Git SHA: %s", git_sha)
        git_dir = abspath(dirname(__file__))
        git_status = subprocess.check_output(["git", "status", "--short"], cwd=git_dir, universal_newlines=True).strip()
        with open(join(args.output_dir, "log", "git_info.json"), "w") as writer:
            json.dump(
                {
                    "branch": git_branch_name,
                    "is_dirty": bool(git_status),
                    "status": git_status,
                    "sha": git_sha,
                },
                writer,
                indent=4,
            )
    except subprocess.TimeoutExpired as e:
        LOGGER.exception(e)
        LOGGER.warn("Git info not found. Moving right along...")
