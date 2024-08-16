import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import timm
import transformers
from torch.utils.data import Dataset

from mafed.data.vl_pythia_pretrain_dataset import PretrainDataset
from mafed.utils.vl_pythia import Collate, DatasetPadding, DatasetSplits, compute_trainable_params
from mafed.model.vl_pythia import VLCLIPGPTNeoXForCausalLM
from mafed.trainer.hf import HuggingfaceTrainer
from mafed.utils.logger import LOGGER


@dataclass
class ModelArguments:
    """Model arguments."""

    model_name: str = field(default="EleutherAI/pythia-410m")
    vision_encoder_name: Literal[
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14-336",
        "timm/eva02_large_patch14_clip_224",
        "timm/eva02_large_patch14_clip_336",
    ] = field(default="openai/clip-vit-base-patch32")

    # These are only used from the CLIP model
    select_layer: int = field(default=-2)
    select_feature: str = field(default="patch")

    tokenizer_name: str = field(default="EleutherAI/pythia-410m")
    tokenizer_truncation_side: Literal["right", "left"] = field(default="right")
    tokenizer_padding_side: Literal["right", "left"] = field(default="right")
    tokenizer_add_special_tokens: bool = field(default=True)
    model_max_length: int = field(default=100)


@dataclass
class DataArguments:
    """Data arguments."""

    dataset_path: str = field(default="contvqa/src/data/contvqa")
    dataset_cache_dir: str = field(default="../datasets/vl_pythia")
    root_dataset_path: str = field(default="../datasets/vl_pythia")
    train_dataset_subset: str = field(default="pretrain")
    eval_dataset_subset: str = field(default="pretrain")


@dataclass
class TrainArgs(transformers.TrainingArguments):
    """Training arguments."""

    output_dir: str = field(default="storage/pretrain-pythia-410m")
    per_device_train_batch_size: int = field(default=128)  # noqa: WPS432
    per_device_eval_batch_size: int = field(default=128)  # noqa: WPS432
    gradient_accumulation_steps: int = field(default=1)
    logging_steps: int = field(default=1)
    save_strategy: str = field(default="steps")
    save_steps: float = field(default=0.1)
    num_train_epochs: int = field(default=2)
    learning_rate: float = field(default=2e-5)  # noqa: WPS432
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="linear")
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    # deepspeed: str = field(default="configs/trainer/zero2.json")
    save_total_limit: int = field(default=2)
    load_best_model_at_end: bool = field(default=True)
    log_level: str = field(default="info")
    save_safetensors: bool = field(default=True)
    evaluation_strategy: str = field(default="steps")
    eval_steps: float = field(default=0.1)
    seed: int = field(default=12345)  # noqa: WPS432
    data_seed: int = field(default=12345)  # noqa: WPS432
    dataloader_num_workers: int = field(default=4)
    logging_nan_inf_filter: bool = field(default=False)
    run_name: str = field(default="pretrain-pythia-410m-fix-center-crop")
    project_name: str = field(default="cl-pretrain-vl-pythia")


def build_model(model_args: ModelArguments) -> VLCLIPGPTNeoXForCausalLM:  # noqa: WPS231
    """Build model."""
    LOGGER.info(f"Building model: {model_args.model_name}")
    model = VLCLIPGPTNeoXForCausalLM.from_pretrained(
        pretrained_model_name=model_args.model_name,
        vision_encoder_name=model_args.vision_encoder_name,
        select_layer=model_args.select_layer,
        select_feature=model_args.select_feature,
        use_flash_attention_2=True,
    )

    # Freeze the vision encoder
    LOGGER.info("Freezing vision encoder")
    for p in model.vision_encoder.parameters():
        p.requires_grad = False

    compute_trainable_params(model)
    return model


def build_tokenizer(model_args: ModelArguments) -> transformers.AutoTokenizer:
    """Build tokenizer."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    # Padding and eos tokens are the same
    tokenizer.eos_token = "<|endoftext|>"  # noqa: S105
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = model_args.model_max_length
    tokenizer.truncation_side = model_args.tokenizer_truncation_side
    tokenizer.padding_side = model_args.tokenizer_padding_side

    return tokenizer


def build_datasets(
    tokenizer: transformers.AutoTokenizer,
    image_preprocessor: transformers.AutoProcessor,
    data_args: DataArguments,
) -> tuple[Dataset, Optional[Dataset], Collate]:
    """Build datasets."""
    LOGGER.info(f"Building datasets: {data_args.train_dataset_subset} {data_args.eval_dataset_subset}")
    train_dataset = PretrainDataset(
        dataset_path=data_args.dataset_path,
        dataset_cache_dir=data_args.dataset_cache_dir,
        root_dataset_path=data_args.root_dataset_path,
        dataset_split=DatasetSplits.TRAIN,
        dataset_subset=data_args.train_dataset_subset,
        tokenizer=tokenizer,
        image_preprocessor=image_preprocessor,
        model_max_length=tokenizer.model_max_length,
    )

    eval_dataset = PretrainDataset(
        dataset_path=data_args.dataset_path,
        dataset_cache_dir=data_args.dataset_cache_dir,
        root_dataset_path=data_args.root_dataset_path,
        dataset_split=DatasetSplits.VALIDATION,
        dataset_subset=data_args.train_dataset_subset,
        tokenizer=tokenizer,
        image_preprocessor=image_preprocessor,
        model_max_length=tokenizer.model_max_length,
    )

    collate_fn = Collate(
        padding=DatasetPadding(input_ids=tokenizer.pad_token_id),
        padding_side=tokenizer.padding_side,
    )
    return train_dataset, eval_dataset, collate_fn


def train():
    """Train."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainArgs))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    # Build model and tokenizer
    model = build_model(model_args=model_args)
    tokenizer = build_tokenizer(model_args=model_args)

    # use the hugginface clip image processor
    if isinstance(model.vision_encoder, transformers.CLIPVisionModel):
        image_preprocessor = transformers.CLIPImageProcessor.from_pretrained(model_args.vision_encoder_name)
    else:
        data_cfg = timm.data.resolve_data_config(model.vision_encoder.pretrained_cfg)
        image_preprocessor = timm.data.create_transform(**data_cfg)

    # Set the environment variables
    # https://docs.wandb.ai/guides/integrations/huggingface
    if train_args.project_name is not None:
        os.environ["WANDB_PROJECT"] = train_args.project_name

    train_dataset, eval_dataset, data_collator = build_datasets(
        tokenizer=tokenizer,
        image_preprocessor=image_preprocessor,
        data_args=data_args,
    )

    trainer = HuggingfaceTrainer(
        args=train_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    train()
