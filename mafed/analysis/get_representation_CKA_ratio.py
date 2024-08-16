"""
Create the plot of the ratio of Text-to-Image CKA Similarity. 

This script assumes that we have run get_average_representation_CKA_per_layer and cached the representations.
"""

import os
from matplotlib import pyplot as plt
import pickle
import numpy as np
from mafed.utils.logger import LOGGER

plt.rcParams.update({"font.size": 14})
colors = ["#FAACC7", "#E05A7B", "#BF132F", "#880d1e"]
NUM_TASKS = 5


def main():
    """Main code."""
    exps = ["Diverse Content", "Taxonomy Content", "Question Types"]
    models = ["VL-Pythia"]
    fig, ax = plt.subplots(len(models), len(exps))
    for row, model in enumerate(models):
        paths = [
            f"storage/results/analysis_plots/CKA/{model}_diverse_cka.pkl",
            f"storage/results/analysis_plots/CKA/{model}_taxonomy_cka.pkl",
            f"storage/results/analysis_plots/CKA/{model}_question_types_cka.pkl",
        ]

        for plot_idx, (exp_name, cached_file) in enumerate(zip(exps, paths)):
            if not os.path.exists(cached_filepath):
                LOGGER.info(f"Cached representations not found in {cached_filepath}:\nPlease run `get_average_CKA_per_layer.py` first to extract the CKA similarities!")
            
            # CKA similarity per layer: Num Checkpoints X (Num Tasks - 1)
            # Num Checkpoints refers to the number of different runs that we take the average of
            # Num Tasks refers to the number of CL tasks in the sequence
            cka = pickle.load(open(cached_filepath, "rb"))
            LOGGER.info(f"Preparing the plots for {model}-{exp_name}")

            num_layers = len([k for k in cka.keys() if "image" in k])
            ax[row, plot_idx].set_title(f"{model}: {exp_name}")
            ax[row, plot_idx].set_xlabel("Layer")
            ax[row, plot_idx].set_xticks(np.arange(1, num_layers + 1))
            if plot_idx == 0:
                ax[row, plot_idx].set_ylabel("T/I CKA Ratio")

            for run in range(NUM_TASKS - 1):
                image_cka = [(cka[f"image:{idx}"][:, run]).mean() for idx in range(1, num_layers + 1)]
                text_cka = [(cka[f"text:{idx}"][:, run]).mean() for idx in range(1, num_layers + 1)]
                ratio_cka = [txt / img for img, txt in zip(image_cka, text_cka)]
                ax[row, plot_idx].plot(
                    list(range(1, num_layers + 1)), ratio_cka, "o--", color=colors[run], label=f"Task {run+2}"
                )
            ax[row, plot_idx].grid()
            ax[row, plot_idx].set_ylim(0.9, 3)
            ax[row, plot_idx].set_yticks(np.arange(1, 3.1, 0.5))

    fig.set_size_inches(14, 9)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, ncols=8, loc="lower center", bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.savefig("CKARatioJoint.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
