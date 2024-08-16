import argparse
import itertools
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import List, Optional

from tqdm import tqdm


@dataclass
class VQA_Answer:
    answer: str
    answer_confidence: str
    answer_id: int


@dataclass
class VQA_Annotation:
    image_id: int
    id: str
    question_id: int
    question: str
    img_fname: str
    multiple_choice_answer: str
    answers: List[VQA_Answer]
    answer_type: str
    question_type: Optional[str] = None


question_task_ids = [
    "contvqa/data/diverse_domains",
    "contvqa/data/question_types",
    "contvqa/data/taxonomy_domains",
]


class VQAInstanceCreator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.annotations_paths = {
            "train": os.path.join(self.data_dir, "v2_mscoco_train2014_annotations.json"),
            "val": os.path.join(self.data_dir, "v2_mscoco_val2014_annotations.json"),
        }
        self.questions_paths = {
            "train": os.path.join(self.data_dir, "v2_OpenEnded_mscoco_train2014_questions.json"),
            "val": os.path.join(self.data_dir, "v2_OpenEnded_mscoco_val2014_questions.json"),
        }

    def read_vqa_annotations(self, file_path):
        with open(file_path, "r") as fp:
            annotations = json.load(fp)["annotations"]
        return annotations

    def read_vqa_questions(self, file_path):
        with open(file_path, "r") as fp:
            questions = {q["question_id"]: q["question"] for q in json.load(fp)["questions"]}
        return questions

    def prepare_vqa_annotations(self, questions_file, answers_file):
        questions = self.read_vqa_questions(questions_file)
        annotations = self.read_vqa_annotations(answers_file)
        vqa_annotations = {}
        for annotation in tqdm(annotations, desc="Processing annotations"):
            qid = str(annotation["question_id"])
            split_name = os.path.basename(answers_file).split(".")[0].split("_")[-2]
            image = f"coco_{split_name}_{str(annotation['image_id']).zfill(12)}"
            vqa_annotations[qid] = VQA_Annotation(
                image_id=annotation["image_id"],
                id=qid,
                question_id=annotation["question_id"],
                question=questions[annotation["question_id"]],
                img_fname=image,
                question_type=annotation.get("question_type"),
                multiple_choice_answer=annotation["multiple_choice_answer"],
                answers=annotation["answers"],
                answer_type=annotation["answer_type"],
            )

        return vqa_annotations

    def get_ids_per_split(self):
        ids_per_split = defaultdict(list)
        for split in ["train", "val", "test"]:
            print(f"Getting ids for {split} split")
            for root_dir in question_task_ids:
                if split == "val":
                    split_file = os.path.join(self.data_dir, root_dir, f"valid_question_ids.json")
                else:
                    split_file = os.path.join(self.data_dir, root_dir, f"{split}_question_ids.json")
                print(f"Reading {split_file}...")
                with open(split_file, "r") as fp:
                    splits_ids = json.load(fp)
                ids_per_split[split].extend(
                    list(itertools.chain.from_iterable([splits_ids[t] for t in splits_ids]))
                )

        ids_per_split = {k: list(set(v)) for k, v in ids_per_split.items()}
        return ids_per_split

    def get_annotations_per_split(self, annotations, ids_per_split):
        annotations_per_split = {}
        for split in ids_per_split.keys():
            print(f"Getting annotations for {split} split")
            annotations_per_split[split] = {qid: asdict(annotations[qid]) for qid in ids_per_split[split]}
        return annotations_per_split

    def run(self):
        annotations = {}
        for split in ["train", "val"]:
            annotations.update(self.prepare_vqa_annotations(self.questions_paths[split], self.annotations_paths[split]))
        ids_per_split = self.get_ids_per_split()

        annotations_per_split = self.get_annotations_per_split(annotations, ids_per_split)
        for split, anns in annotations_per_split.items():
            print(f"Saving {len(anns)} annotations for {split} split")
            with open(os.path.join(self.data_dir, f"{split}_annotations.json"), "w") as fp:
                json.dump(anns, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="storage/data/VQA", help="Data root dir")
    args = parser.parse_args()
    VQAInstanceCreator(args.data_dir).run()
