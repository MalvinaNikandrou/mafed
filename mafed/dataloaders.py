import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from mafed.data import PrefetchLoader, collate_fn, datasets_map
from mafed.utils.logger import LOGGER


def get_task_question_ids_file(question_task_ids_dir, exp_name, split):
    split = "valid" if split == "val" else split
    return os.path.join(question_task_ids_dir, exp_name, f"{split}_question_ids.json")


def get_task_dataloader(
    task,
    config,
    batch_size,
    num_workers,
    img_dirs,
    split,
    tokenizer,
    image_preprocessor,
) -> dict[str, DataLoader]:
    """Prepare validation dataloaders for all tasks once."""

    datasets = []

    if isinstance(img_dirs, dict):
        task_img_dirs = img_dirs[task]
    else:
        task_img_dirs = img_dirs

    LOGGER.debug(f"Start dataset {task}")
    for img_dir in task_img_dirs:
        datasets.append(
            datasets_map["valid"][config.model_type](
                image_dir=img_dir,
                data_path=config.data_dir,
                split_file=get_task_question_ids_file(config.question_task_ids, config.exp, split),
                task=task,
                tokenizer=tokenizer,
                image_preprocessor=image_preprocessor,
                split=split,
            )
        )
    LOGGER.debug(f"End dataset {task}")
    dataloader = PrefetchLoader(
        DataLoader(
            ConcatDataset(datasets),
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=config.pin_mem,
            collate_fn=collate_fn["valid"][config.model_type],
        )
    )
    LOGGER.debug(f"End dataloader {task}")

    return dataloader


def get_val_dataloaders(config, tokenizer, image_preprocessor, data_split="valid") -> dict:
    """Prepare validation dataloaders for all tasks once."""
    if data_split == "valid":
        img_dirs = config.val_img_dirs
    elif data_split == "test":
        img_dirs = config.test_img_dirs

    LOGGER.info(f"Loading Validation Datasets")
    val_dataloaders = {}

    for task in config.tasks:
        val_dataloaders[task] = get_task_dataloader(
            task=task,
            config=config,
            batch_size=config.val_batch_size,
            num_workers=config.n_workers,
            img_dirs=img_dirs,
            split="val",
            tokenizer=tokenizer,
            image_preprocessor=image_preprocessor,
        )

    return val_dataloaders


def prepare_train_dataset(config, task, tokenizer=None, image_preprocessor=None):
    # Load the training data
    train_datasets = []
    if isinstance(config.train_img_dirs, dict):
        img_dirs = config.train_img_dirs[task]
    else:
        img_dirs = config.train_img_dirs

    for img_dir in img_dirs:
        train_datasets.append(
            datasets_map["train"][config.model_type](
                image_dir=img_dir,
                data_path=config.data_dir,
                split_file=get_task_question_ids_file(config.question_task_ids, config.exp, "train"),
                task=task,
                split="train",
                randaug=True,
                tokenizer=tokenizer,
                image_preprocessor=image_preprocessor,
            )
        )
    return ConcatDataset(train_datasets)


class TaskDataModule(LightningDataModule):
    """Datamodule for the current training task."""

    def __init__(self, config, task, val_dataloaders, tokenizer=None, image_preprocessor=None) -> None:
        super().__init__()
        self.task = task
        self.config = config
        self._val_dataset = val_dataloaders[task].dataset
        self._num_workers = int(self.config.n_workers)
        self.model_type = config.model_type

        self._image_preprocessor = image_preprocessor
        self._tokenizer = tokenizer

    def setup(self, stage):
        # Load the training data
        self._train_dataset = prepare_train_dataset(
            self.config, task=self.task, tokenizer=self._tokenizer, image_preprocessor=self._image_preprocessor
        )

    def train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(
            self._train_dataset,
            num_workers=self._num_workers,
            batch_size=self.config.batch_size,
            pin_memory=self.config.pin_mem,
            collate_fn=collate_fn["train"][self.model_type],
        )

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            num_workers=self.config.n_workers,
            batch_size=self.config.val_batch_size,
            pin_memory=self.config.pin_mem,
            collate_fn=collate_fn["valid"][self.model_type],
        )


class MultitaskDataModule(LightningDataModule):
    """Datamodule for the all training tasks."""

    def __init__(self, config, tokenizer, image_preprocessor) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        # load DBs and image dirs
        self._num_workers = int(self.config.n_workers)
        self._image_preprocessor = image_preprocessor
        self._tokenizer = tokenizer

    def train_dataloader(self) -> DataLoader:
        # Load the training data
        train_img_dirs = []
        if isinstance(self.config.train_img_dirs, dict):
            for task in self.config.train_img_dirs.keys():
                train_img_dirs.extend(self.config.train_img_dirs[task])
        elif isinstance(self.config.train_img_dirs, list):
            train_img_dirs = self.config.train_img_dirs

        train_datasets = []
        for img_dir in train_img_dirs:
            train_datasets.append(
                datasets_map["train"][self.model_type](
                    image_dir=img_dir,
                    data_path=self.config.data_dir,
                    split_file=get_task_question_ids_file(self.config.question_task_ids, self.config.exp, "train"),
                    task="joint",
                    split="train",
                    randaug=True,
                    tokenizer=self._tokenizer,
                    image_preprocessor=self._image_preprocessor,
                )
            )
        train_dataset = ConcatDataset(train_datasets)
        LOGGER.info(f"Loaded {len(train_dataset)} train samples")
        train_dataloader = DataLoader(
            train_dataset,
            num_workers=self._num_workers,
            batch_size=self.config.batch_size,
            pin_memory=self.config.pin_mem,
            collate_fn=collate_fn["train"][self.model_type],
            shuffle=True,
        )

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_img_dirs = []
        if isinstance(self.config.val_img_dirs, dict):
            for task in self.config.val_img_dirs.keys():
                val_img_dirs.extend(self.config.val_img_dirs[task])
        elif isinstance(self.config.val_img_dirs, list):
            val_img_dirs = self.config.val_img_dirs

        val_datasets = []
        for img_dir in val_img_dirs:
            val_datasets.append(
                datasets_map["valid"][self.model_type](
                    image_dir=img_dir,
                    data_path=self.config.data_dir,
                    split_file=get_task_question_ids_file(self.config.question_task_ids, self.config.exp, "val"),
                    task="joint",
                    split="val",
                    tokenizer=self._tokenizer,
                    image_preprocessor=self._image_preprocessor,
                )
            )
        val_dataset = ConcatDataset(val_datasets)
        LOGGER.info(f"Loaded {len(val_dataset)} valid samples")
        val_dataloader = DataLoader(
            val_dataset,
            num_workers=self._num_workers,
            batch_size=self.config.val_batch_size,
            pin_memory=self.config.pin_mem,
            collate_fn=collate_fn["valid"][self.model_type],
        )

        return val_dataloader
