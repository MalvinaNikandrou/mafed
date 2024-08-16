from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional, Union

import datasets
import torch
from PIL import Image
from pydantic import BaseModel
from torch.nn.utils.rnn import pad_sequence

from mafed.utils.logger import LOGGER


DatasetFeatures = datasets.Features(
    {
        "image": datasets.Image(),
        "source": datasets.Value("string"),
        "caption": datasets.Value("string"),
        # We commit any kind of additional information in json format in `meta`
        "metadata": datasets.Value("string"),
    }
)


BASE_DIR = Path("storage/datasets")


class DatasetPaths:  # noqa: WPS230
    """Dataclass for data paths.

    Some datasets are downloaded with a different tool (img2dataset, etc), or have separate
    hugginface cache (visual genome). We need to know where they are.
    """

    def __init__(self, base_dir: Union[str, Path] = BASE_DIR) -> None:
        self.storage = Path(base_dir)

        self.coco_cache_dir = self.storage.joinpath("coco")
        self.visual_genome_cache_dir = self.storage.joinpath("visual_genome")

        # These datasets are downloaded with an external tool aka img2dataset
        self.conceptual_captions_cache_dir = self.storage.joinpath("conceptual_captions")
        self.sbu_captions_cache_dir = self.storage.joinpath("sbu_captions")


class DatasetSplits(datasets.Split):  # type: ignore[misc]
    """Dataset splits."""

    @classmethod
    def list_splits(cls) -> list[datasets.Split]:
        """List all splits."""
        return [
            cls.TRAIN,
            cls.VALIDATION,
            cls.TEST,
        ]


class PretrainInstance(BaseModel):
    """Dataset instance."""

    image: Image.Image
    caption: str
    source: str
    metadata: str

    class Config:
        """Updated config."""

        arbitrary_types_allowed = True


class Img2DatasetModel(BaseModel):
    """Img2Dataset metadata."""

    caption: str
    url: str
    key: str
    shard_id: str
    status: str
    width: Optional[float] = None
    height: Optional[float] = None
    original_width: Optional[float] = None
    original_height: Optional[float] = None
    exif: Optional[str] = None
    sha256: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        """Check if the instance was successfully downloaded."""
        return self.status == "success"


@dataclass
class PretrainDatasetItem:
    """Output for the dataset reader."""

    input_ids: torch.Tensor
    pixel_values: torch.Tensor = None
    labels: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    raw_target: Optional[dict[str, Any]] = None


class DatasetNames(Enum):
    """Dataset names."""

    cc3m = "conceptual_captions_3m"
    coco = "coco"
    visual_genome = "visual_genome"
    sbu_captions = "sbu_captions"
    pretrain = "pretrain"

    @classmethod
    def list_all_dataset_names(cls) -> list[str]:
        """List all dataset names."""
        return [name.value for name in cls]


@dataclass
class PretrainDatasets:
    """A dataclass that contains all datasets that are used for pretraining."""

    pretrain_dataset_map: Mapping[str, bool] = MappingProxyType(
        {
            DatasetNames.cc3m.value: True,
            DatasetNames.coco.value: True,
            DatasetNames.visual_genome.value: True,
            DatasetNames.sbu_captions.value: True,
        }
    )

    def __init__(self) -> None:
        """Verify that all datasets are in the map."""
        if len(self.pretrain_dataset_map) != len(DatasetNames) - 1:
            raise AssertionError("Pretrain dataset map is not equal to all datasets!")

        for key, _ in self.pretrain_dataset_map.items():
            missing = key not in DatasetNames.list_all_dataset_names() and key != DatasetNames.pretrain.value
            if missing:
                raise ValueError(f"Dataset {key} not in {DatasetNames.list_all_dataset_names()}!")

    def is_in_pretrain_dataset(self, dataset_name: str) -> bool:
        """Check if the dataset is in the pretrain dataset."""
        return self.pretrain_dataset_map[dataset_name]

    def list_all_pretrain_dataset_names(self) -> list[str]:
        """List all pretrain datasets."""
        return [name for name, is_in_pretrain in self.pretrain_dataset_map.items() if is_in_pretrain]


def _pad_sequence(
    seq: list[torch.Tensor],
    padding_value: int,
    padding_side: str = "right",
) -> torch.Tensor:
    """Pad a sequence of tensors.

    IMPORTANT: Use padding_side="left" to pad on the left side when dealing with batch generation.
    """
    if not seq:
        return torch.empty(0)

    if padding_side == "right":
        return pad_sequence(seq, batch_first=True, padding_value=padding_value)
    rev_seq = [s.flip(0) for s in seq]
    rev_padded = pad_sequence(rev_seq, batch_first=True, padding_value=padding_value)
    return rev_padded.flip(-1)


@dataclass
class DatasetPadding:
    """Padding values used by collate."""

    input_ids: int = 0
    pixel_values: int = 0
    attention_mask: int = 0
    labels: int = -100


@dataclass
class DatasetItemCollateFn:
    """Used to determine what to do in the collate function for element in an example."""

    input_ids = "pad"
    labels = "pad"
    attention_mask = "pad"
    pixel_values = "stack"
    raw_target = "raw"


@dataclass
class Collate:
    """Collate class.

    This is a class to ensure that the padding values are correctly passed when creating the batch.
    """

    padding: DatasetPadding
    padding_side: str = "right"
    collate_mode = DatasetItemCollateFn()

    def __call__(self, batch: list[PretrainDatasetItem]) -> dict[str, Any]:
        """Collate lists of samples into batches after padding."""
        data_fields = fields(PretrainDatasetItem)

        raw_batch: dict[Any, Any] = {}
        for field in data_fields:
            field_mode = getattr(self.collate_mode, field.name)
            if field_mode == "raw":
                raw_batch[field.name] = self._process_raw_field(field.name, batch)
            elif field_mode == "stack":
                raw_batch[field.name] = self._process_stack_field(field.name, batch)
            elif field_mode == "pad":
                raw_batch[field.name] = self._process_pad_field(field.name, batch)
        return raw_batch

    def _process_raw_field(self, field_name: str, batch: list[PretrainDatasetItem]) -> list[Any]:
        """Raw fields do not require any processing, just return the list of raw items."""
        return [
            getattr(sample, field_name)
            for sample in batch
            if sample is not None and getattr(sample, field_name) is not None
        ]

    def _process_stack_field(self, field_name: str, batch: list[PretrainDatasetItem]) -> Optional[torch.Tensor]:
        """Fields that do not require any padding (ie pixel values) can be stacked."""
        sequence = [
            getattr(sample, field_name)
            for sample in batch
            if sample is not None and getattr(sample, field_name) is not None
        ]
        if not sequence:
            return None
        return torch.stack(sequence)

    def _process_pad_field(self, field_name: str, batch: list[PretrainDatasetItem]) -> Optional[torch.Tensor]:
        """Fields that require padding (ie input_token_ids) need to be padded."""
        sequence = [
            getattr(sample, field_name)
            for sample in batch
            if sample is not None and getattr(sample, field_name) is not None
        ]
        if not sequence:
            return None

        return _pad_sequence(
            seq=sequence,
            padding_value=getattr(self.padding, field_name),
            padding_side=self.padding_side,
        )


def compute_trainable_params(model: torch.nn.Module) -> None:
    """Compute trainable parameters."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    train_params = sum([p.numel() for p in model_parameters])
    LOGGER.info(f"{sum([p.numel() for p in model.parameters()])} params and {train_params} trainable params")
