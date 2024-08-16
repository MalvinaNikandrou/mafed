import os
from typing import Union

import torch
from PIL import Image
from toolz.sandbox import unzip
from torchvision.transforms import Compose
from transformers import CLIPImageProcessor, PreTrainedTokenizer

from mafed.data.base_data import BaseDataset
from mafed.data.vqa_utils import normalize_answer
from mafed.utils.vl_pythia import PretrainDatasetItem, _pad_sequence


def get_image_path(image_dir, image_name):
    """Get image path from the image db fname."""
    if image_name.startswith("coco"):
        image_fname_fields = os.path.splitext(image_name)[0].split("_")
        image_path = f"COCO_{image_fname_fields[1]}_{image_fname_fields[2]}.jpg"
    elif "abstract" in image_name:
        image_path = f"{image_name.split('.npz')[0]}.png"
    elif "VizWiz" in image_name:
        image_path = f"{image_name.split('.npz')[0]}.jpg"
    else:
        image_path = image_name
    image_path = os.path.join(image_dir, image_path)
    return image_path


class VLPythiaVQADataset(BaseDataset):
    """VQA dataset."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        image_preprocessor: Union[CLIPImageProcessor, Compose],
        image_dir=None,
        data_path=None,
        split_file="splits.json",
        task=None,
        split=None,
        **kwargs,
    ) -> None:
        """
        tokenizer (PreTrainedTokenizer): Text tokenizer.
        image_preprocessor (CLIPImageProcessor, Compose): Image preprocessor.
        image_dir (str, optional): Path to images.
        data_path (str, optional): Path to VQA annotations.
        split_file (str, optional): Path to file containing the question ids for each task and split.
        task (str, optinal): Use only the samples that belong to this task.
        split (str, optional): [train, val] Use only the samples that belong to this split.
        """
        super().__init__(
            data_path=data_path,
            split_file=split_file,
            task=task,
            split=split,
        )

        self.tokenizer = tokenizer
        self.image_preprocessor = image_preprocessor
        self._return_tensors = "pt"
        self.image_dir = image_dir
        self.split = split

    def _prepare_image(self, image_name):
        image_path = get_image_path(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if isinstance(self.image_preprocessor, CLIPImageProcessor):
            return self.image_preprocessor.preprocess(image, return_tensors="pt").pixel_values.squeeze(0)
        return self.image_preprocessor(image).squeeze(0)

    def _get_input_ids(self, question, answer=None):
        input_ids = self.tokenizer(question).input_ids
        if self.split == "train":
            labels = [-100 for _ in input_ids]
            answer_encoding = self.tokenizer(answer).input_ids
            answer_encoding.append(self.tokenizer.eos_token_id)
            # ADD SPECIAL EOS TOKEN
            input_ids.extend(answer_encoding)
            labels.extend(answer_encoding)
            return torch.tensor(input_ids), torch.tensor(labels)
        return torch.tensor(input_ids), None

    def __getitem__(self, index: int) -> PretrainDatasetItem:
        """Get a single instance from the dataset."""
        example = super().__getitem__(index)
        pixel_values = self._prepare_image(example["img_fname"])
        question = self.format_text(example["question"])
        answers = [normalize_answer(ans["answer"]) for ans in example["answers"]]

        normalized_answer = normalize_answer(example["multiple_choice_answer"])
        answer = self.format_text(normalized_answer, capitalize=False)
        text_encoding, labels = self._get_input_ids(question, answer)
        attention_mask = torch.ones_like(text_encoding)
        return (
            text_encoding,
            attention_mask,
            pixel_values,
            labels,
            example["img_fname"],
            answers,
            example["question_id"],
            {"question": question, "answer": answer},
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

        add_fullstop = punctuate and not text.endswith((".", "?", "!"))
        if add_fullstop:
            text = f"{text}."

        return text


def vlpythia_vqa_collate(inputs):
    (
        input_ids,
        attention_mask,
        pixel_values,
        labels,
        img_names,
        answers,
        qids,
        raw_target,
    ) = map(list, unzip(inputs))

    pixel_values = torch.stack(pixel_values)
    input_ids = _pad_sequence(input_ids, padding_side="left", padding_value=0)
    attention_mask = _pad_sequence(attention_mask, padding_side="left", padding_value=0)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_name": img_names,
        "answers": answers,
        "qids": qids,
        "raw_target": raw_target,
    }

    if labels[0] is not None:
        labels = _pad_sequence(labels, padding_side="left", padding_value=-100)
        batch["labels"] = labels

    return batch
