"""
Compute the CKA similarity between the representations of the different tasks.

Basic use:
python mafed/analysis/get_average_representation_CKA_per_layer.py \
    --config config/train-vqa-base-cl-local-vlpythia.json \
    --exp question_types  \
    --output_file storage/results/analysis_plots/CKA/vlpythia_qtypes_cka \
    --checkpoint_dirs storage/outputs/question_types/run1/vl-pythia-eva-410m_bsz128_lr5e-5_naive/vl-pythia-eva-410m_bsz128_lr5e-5_naive/ckpt/ \
        storage/outputs/question_types/run2/vl-pythia-eva-410m_bsz128_lr5e-5_naive/vl-pythia-eva-410m_bsz128_lr5e-5_naive/ckpt/ \
        storage/outputs/question_types/run3/vl-pythia-eva-410m_bsz128_lr5e-5_naive/vl-pythia-eva-410m_bsz128_lr5e-5_naive/ckpt/ 
"""

import argparse
import json
import os
from os.path import join
from matplotlib import pyplot as plt
import pickle
import torch
import transformers
import timm
from tqdm import tqdm
import numpy as np

from pytorch_lightning import seed_everything
from mafed.dataloaders import get_val_dataloaders
from mafed.utils.logger import LOGGER
from mafed.utils.misc import parse_with_config
from mafed.utils.cka import feature_space_linear_cka
from mafed.utils.checkpoint import load_model_from_checkpoint
from mafed.utils.eval_utils import get_checkpoint_path
from mafed.pretrain_vlpythia import ModelArguments, build_tokenizer


plt.rcParams.update({"font.size": 12})


class CKARepresentationSimilarity:

    def __init__(self, config):
        self.config = config
        self.image_len = 256  # Number of image tokens
        self.max_samples = 500000
        self.titles_map = {
            "diverse_domains": "Diverse Content",
            "taxonomy_domains": "Taxonomy Content",
            "question_types": "Question Types",
        }

    def plot_similarities(self, cka):
        LOGGER.info("Preparing the plots")
        plt.figure()
        # Get title
        exp_name = self.titles_map[selg.config.exp]
        plt.title(f"VL-Pythia {exp_name}")

        num_layers = len([k for k in cka.keys() if "image" in k])
        x = list(range(1, num_layers + 1))
        plt.xlabel("Layer")
        plt.xticks(np.arange(1, num_layers + 1))
        # Image Similarity
        y = [cka[f"image:{idx}"][:, 0].mean() for idx in range(1, num_layers + 1)]
        plt.plot(x, y, "s--", color="#8b0000", label="Task 2 (V)")
        y = [cka[f"image:{idx}"][:, 1].mean() for idx in range(1, num_layers + 1)]
        plt.plot(x, y, "s--", color="#dc143c", label="Task 3 (V)")
        y = [cka[f"image:{idx}"][:, 2].mean() for idx in range(1, num_layers + 1)]
        plt.plot(x, y, "s--", color="#f08080", label="Task 4 (V)")
        y = [cka[f"image:{idx}"][:, 3].mean() for idx in range(1, num_layers + 1)]
        plt.plot(x, y, "s--", color="#ffa07a", label="Task 5 (V)")
        # Text Similarity
        y = [cka[f"text:{idx}"][:, 0].mean() for idx in range(1, num_layers + 1)]
        plt.plot(x, y, "o--", color="midnightblue", label="Task 2 (T)")
        y = [cka[f"text:{idx}"][:, 1].mean() for idx in range(1, num_layers + 1)]
        plt.plot(x, y, "o--", color="#0047AB", label="Task 3 (T)")
        y = [cka[f"text:{idx}"][:, 2].mean() for idx in range(1, num_layers + 1)]
        plt.plot(x, y, "o--", color="#6495ED", label="Task 4 (T)")
        y = [cka[f"text:{idx}"][:, 3].mean() for idx in range(1, num_layers + 1)]
        plt.plot(x, y, "o--", color="#89CFF0", label="Task 5 (T)")
        plt.ylim(0.5, 1.01)
        plt.tight_layout()
        plt.legend()
        plt.grid()
        print(f"{self.config.output_file}_image_text.pdf")
        plt.savefig(f"{self.config.output_file}_image_text.pdf", bbox_inches="tight")

    @torch.no_grad()
    def get_representations(
        self,
        model,
        val_dataloader,
        qid2idx,
        hidden_state_dim=768,
    ):
        """Get hidden states per sample."""
        model.eval()
        num_layers = len(model.gpt_neox.layers)
        self.all_layers = [f"image:{i+1}" for i in range(num_layers)] + [f"text:{i+1}" for i in range(num_layers)]
        results = {
            layer: np.zeros((min(self.max_samples, len(qid2idx)), hidden_state_dim)) for layer in self.all_layers
        }
        LOGGER.info(f"Num samples = {min(self.max_samples, len(qid2idx))}")
        for batch in tqdm(val_dataloader, desc="Extracting representations"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch, output_hidden_states=True, return_dict=True, compute_loss=False)

            for within_batch_index, qid in enumerate(batch["qids"]):
                results_index = qid2idx[qid]
                txt_len = batch["attention_mask"][within_batch_index].sum().item()
                for layer in range(1, num_layers + 1):
                    hidden_states = outputs.hidden_states[layer][within_batch_index]
                    # Get the average question representation
                    txt_emb = hidden_states[-txt_len:].mean(0).cpu().numpy()
                    results[f"text:{layer}"][results_index] = txt_emb

                    # Get the average image representation
                    img_emb = hidden_states[: self.image_len].mean(0).cpu().numpy()
                    results[f"image:{layer}"][results_index] = img_emb

        # Check that hidden states are not zeros
        for layer in self.all_layers:
            res = results[layer].mean(1)
            assert np.all(res != 0)
        return results

    def extract_cka_similarities(self):
        """
        Get the CKA similarities, in the format:
        {
            layer_index: {
                run_index: [cka_values]
        }
        layer_index: the layer from which the representations are extracted
        run_index: the index of the run. We plot the average CKA similarity across runs.
        cka_values: the CKA similarity values for each task
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(self.config.seed)
        with open(join(self.config.checkpoint_dirs[0], "task_order.json"), "r") as fp:
            self.config.tasks = json.load(fp)
        LOGGER.info(f"Batch size = {self.config.batch_size}")
        model = None
        model_args = ModelArguments(
            model_name=self.config.model_name,
            vision_encoder_name=self.config.vision_encoder_name,
            tokenizer_name=self.config.tokenizer_name,
            tokenizer_padding_side="left",
        )
        tokenizer = build_tokenizer(model_args=model_args)
        if "eva" not in model_args.vision_encoder_name:
            image_preprocessor = transformers.CLIPImageProcessor.from_pretrained(model_args.vision_encoder_name)
        else:
            data_cfg = {
                "input_size": (3, 224, 224),
                "interpolation": "bicubic",
                "mean": (0.48145466, 0.4578275, 0.40821073),
                "std": (0.26862954, 0.26130258, 0.27577711),
                "crop_pct": 0.9,
                "crop_mode": "center",
            }
            image_preprocessor = timm.data.create_transform(**data_cfg)
        val_dataloaders = get_val_dataloaders(
            config=self.config,
            data_split="valid",
            tokenizer=tokenizer,
            image_preprocessor=image_preprocessor,
        )

        self.all_layers = model.config.num_hidden_layers
        # CKA similarity per layer: Num Checkpoints X (Num Tasks - 1)
        # Num Checkpoints refers to the number of different runs that we take the average of
        # Num Tasks refers to the number of CL tasks in the sequence
        cka = {layer: np.zeros((len(self.config.checkpoint_dirs), len(self.config.tasks) - 1)) for layer in self.all_layers}
        for run_index, checkpoint_dir in enumerate(self.config.checkpoint_dirs):
            LOGGER.info(f"Run: {run_index}")
            all_results = []
            with open(join(checkpoint_dir, "task_order.json"), "r") as fp:
                tasks = json.load(fp)

            # Get the qid to index mapping to store the representations in the same order
            val_task = tasks[self.config.reference_task]
            qid2idx = {}
            for batch in val_dataloaders[val_task]:
                for qid in batch["qids"]:
                    qid2idx[qid] = len(qid2idx)
            for task_id, task in enumerate(tasks):
                # Initialize for new task
                LOGGER.info(f"*****  Task: {task.upper()} *****")
                best_model = get_checkpoint_path(
                    task_id, task, checkpoint_dir, extension=self.config.checkpoint_extension
                )
                LOGGER.info(f"1: Load the checkpoint: {best_model}")
                # Prepare model
                if model:
                    del model
                model = load_model_from_checkpoint(
                    checkpoint=best_model,
                    model_name=self.config.model_name,
                    vision_encoder_name=self.config.vision_encoder_name,
                    device=device,
                )
                model.eval()
                LOGGER.info("2. Get the representations")
                results = self.get_representations(
                    model=model,
                    val_loader=val_dataloaders[val_task],
                    qid2idx=qid2idx,
                    hidden_state_dim=model.config.hidden_size,
                )
                all_results.append(results)

            # Compute the CKA similarity between the reference task and other task representations
            LOGGER.info("Computing CKA...")
            for layer_index in self.all_layers:
                #
                time_index = 0
                for task_index in range(len(tasks)):
                    if task_index == self.config.reference_task:
                        continue
                    cka[layer_index][run_index][time_index] = feature_space_linear_cka(
                        all_results[task_index][layer_index], all_results[self.config.reference_task][layer_index]
                    )
                    time_index += 1
        return cka

    def run(self):
        cache_file = os.path.join(self.config.output_file + ".pkl")
        if os.path.exists(cache_file):
            cka = pickle.load(open(cache_file, "rb"))
        else:
            cka = self.extract_cka_similarities()
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            pickle.dump(cka, open(cache_file, "wb"))
        self.plot_similarities(cka)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--config",
        default="config/train-vqa-base-cl-local-vlpythia.json",
        type=str,
        help="Json config file for model architecture",
    )
    parser.add_argument(
        "--checkpoint_dirs",
        default=None,
        nargs="+",
        help="Path to a directory of task models",
    )
    parser.add_argument(
        "--checkpoint_extension",
        default=".ckpt",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        type=str,
        help="The output directory where the plots are saved and the representations are cached.",
    )
    # Prepro parameters
    parser.add_argument(
        "--max_txt_len",
        type=int,
        default=60,
        help="max number of tokens in text (BERT BPE)",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Total batch size for validation. " "(batch by tokens)",
    )
    # device parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument(
        "--exp",
        default="question_types",
        choices=[
            "question_types",
            "diverse_domains",
            "taxonomy_domains",
        ],
        help="Experiment name. This should match the [exp]_splits.json file defining the tasks",
    )
    parser.add_argument("--model_name", default="EleutherAI/pythia-410m")
    parser.add_argument("--tokenizer_name", default="EleutherAI/pythia-410m")
    parser.add_argument("--vision_encoder_name", default="timm/eva02_large_patch14_clip_224")
    parser.add_argument("--reference_task", default=0, type=int, help="Task representations to compare against")
    args = parse_with_config(parser)

    CKARepresentationSimilarity.run(args)
