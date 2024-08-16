import json
from typing import Union

import torch
from torchvision.transforms import Compose

from datasets import load_dataset, Split
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, PreTrainedTokenizer

from mafed.utils.vl_pythia import DatasetPaths, PretrainDatasetItem, PretrainInstance
from mafed.utils.boxes import ObjectCenterCrop


class PretrainDataset(Dataset[PretrainDatasetItem]):
    """Pretrain dataset."""

    def __init__(
        self,
        dataset_path: str,
        dataset_cache_dir: str,
        root_dataset_path: str,
        dataset_split: Split,
        tokenizer: PreTrainedTokenizer,
        # we either use a CLIP model or an EVA model
        image_preprocessor: Union[CLIPImageProcessor, Compose],
        dataset_subset: str = "vl_pythia_pretrain",
        model_max_length: int = 500,
    ) -> None:
        dataset_paths = DatasetPaths(base_dir=root_dataset_path)
        self.dataset = load_dataset(
            dataset_path,
            dataset_subset,
            cache_dir=dataset_cache_dir,
            root_dataset_path=root_dataset_path,
            dataset_paths=dataset_paths,
            verification_mode="no_checks",
            trust_remote_code=True,
        )[dataset_split]

        self.dataset_split = dataset_split
        self.tokenizer = tokenizer
        self.image_preprocessor = image_preprocessor
        self.model_max_length = model_max_length
        if isinstance(image_preprocessor, CLIPImageProcessor):
            size = self.image_preprocessor.crop_size["height"]
        else:
            # image_preprocessor is a Compose object
            # image_preprocessor.transforms
            # [
            #     Resize(size=336, interpolation=bicubic, max_size=None, antialias=warn),
            #     CenterCrop(size=(336, 336)),
            #     ToTensor(),
            #     Normalize(mean=tensor([0.4815, 0.4578, 0.4082]), std=tensor([0.2686, 0.2613, 0.2758]))
            # ]
            size = self.image_preprocessor.transforms[0].size

        self._center_crop_transform = ObjectCenterCrop((size, size))

        self._return_tensors = "pt"

    def __len__(self) -> int:
        """Return the total number of instances within the database."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> PretrainDatasetItem:
        """Get a single instance from the dataset."""
        raw_instance = self.dataset[index]
        instance = PretrainInstance.model_validate(raw_instance)
        return self.process_instance(instance)

    def process_instance(self, instance: PretrainInstance) -> PretrainDatasetItem:
        """Process the instance."""
        image = instance.image.convert("RGB")
        metadata = json.loads(instance.metadata)
        if instance.source == "visual_genome":
            image = self._center_crop_transform(image, metadata["bbox"])

        if isinstance(self.image_preprocessor, CLIPImageProcessor):
            visual_encoding = self.image_preprocessor.preprocess(image, return_tensors="pt")
            pixel_values = visual_encoding.pixel_values.squeeze(0)
        else:
            pixel_values = self.image_preprocessor(image).squeeze(0)

        caption = self.format_text(
            instance.caption,
            strip=True,
            punctuate=True,
            capitalize=True,
        )

        text_encoding = self.tokenizer(caption, return_tensors="pt")

        input_ids = text_encoding.input_ids
        # Labels are the same as input_ids, the shift is handled by the model
        labels = text_encoding.input_ids.clone()

        return PretrainDatasetItem(
            input_ids=input_ids.squeeze(0),
            labels=labels.squeeze(0),
            pixel_values=pixel_values,
            attention_mask=torch.ones_like(input_ids).squeeze(0),
            raw_target={
                "caption": caption,
                "metadata": json.loads(instance.metadata),
                # "image": image,
            },
        )

    def format_text(
        self,
        text: str,
        strip: bool = True,
        capitalize: bool = True,
        punctuate: bool = True,
    ) -> str:
        """Format the text."""
        if strip:
            text = text.strip()

        if capitalize:
            text = text.capitalize()

        add_fullstop = punctuate and not text.endswith(".") and not text.endswith("?") and not text.endswith("!")

        if add_fullstop:
            text = f"{text}."

        return text


# from transformers import AutoTokenizer

# dataset_path = "storage/data/vl_pythia"
# dataset_cache_dir = "storage/data/vl_pythia"
# root_dataset_path = "storage/data/vl_pythia"
# dataset_split = "train"
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
# image_preprocessor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
# dataset_subset = "visual_genome"
# model_max_length = 50
# xx = PretrainDataset(
#     dataset_path,
#     dataset_cache_dir,
#     root_dataset_path,
#     dataset_split,
#     tokenizer,
#     image_preprocessor,
#     dataset_subset,
#     model_max_length
# )

# for example in xx:
#     print(example)
#     breakpoint()
