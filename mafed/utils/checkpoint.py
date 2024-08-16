from os.path import join
from typing import Optional

import torch

from mafed.model import VLCLIPGPTNeoXForCausalLM


def load_model_from_checkpoint(
    checkpoint: str,
    device: str,
    model_name: Optional[str] = None,
    vision_encoder_name: Optional[str] = None,
):
    """Load a model from a checkpoint."""
    map_location = {"cuda:0": device.type}
    state_dict = torch.load(checkpoint, map_location=map_location)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {module.replace("model.", ""): params for module, params in state_dict.items()}
    model = VLCLIPGPTNeoXForCausalLM.from_pretrained(
        model_name,
        vision_encoder_name=vision_encoder_name,
        state_dict=state_dict,
        use_flash_attention_2=True,
    )
    model.to(device)
    return model


def get_initialization_checkpoint(opts, task_id=0, checkpoint_extension=".ckpt") -> str:
    """Get a checkpoint path to initialize the model before training for the first task."""
    checkpoint = None
    if task_id == 0:
        if opts.checkpoint is not None:
            checkpoint = opts.checkpoint
        elif opts.checkpoint_dir is not None:
            checkpoint = join(opts.checkpoint_dir, f"{opts.tasks[0]}_best{checkpoint_extension}")

    return checkpoint
