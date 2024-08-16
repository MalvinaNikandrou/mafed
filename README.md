# Enhancing Continual Learning in Visual Question Answering with Modality-Aware Feature Distillation (ALVR 2024)
Code for Modality-Aware Feature Distillation ([MAFED]([https://arxiv.org/abs/2406.19297](https://aclanthology.org/2024.alvr-1.6/))).

## Requirements
Install a conda environment:

```bash
# Clone the  Repository and setup the environment
git clone https://github.com/malvinan/mafed.git
cd mafed

conda create --name mafed_env python=3.9 -y
conda activate mafed_env

pip install -e .
pip install flash-attn --no-build-isolation
```

### Download and preprocess the Data

This includes:
- Download the COCO images from [https://cocodataset.org/#download](https://cocodataset.org/#download)
- Download the VQA-v2 annotations from [https://visualqa.org/download.html](https://visualqa.org/download.html)
- Download the task splits from [https://github.com/MalvinaNikandrou/contvqa](https://github.com/MalvinaNikandrou/contvqa)

```
./scripts/download_data.sh
```

### Download the pretrained VLPythia models
```
python mafed/utils/download_models.py
```

## Usage
To run a finetuning script for the different task orders, have a look at:

```
./scripts/run_finetuning.sh
```

To run different CL methods, have a look at:
```
./scripts/run_seed42.sh
```

Note: The current codebase has not been tested for distributed training.

## How to Cite

```
@inproceedings{nikandrou-etal-2024-enhancing,
    title = "Enhancing Continual Learning in Visual Question Answering with Modality-Aware Feature Distillation",
    author = "Nikandrou, Malvina  and
      Pantazopoulos, Georgios  and
      Konstas, Ioannis  and
      Suglia, Alessandro",
    editor = "Gu, Jing  and
      Fu, Tsu-Jui (Ray)  and
      Hudson, Drew  and
      Celikyilmaz, Asli  and
      Wang, William",
    booktitle = "Proceedings of the 3rd Workshop on Advances in Language and Vision Research (ALVR)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.alvr-1.6",
    pages = "73--85",
    abstract = "Continual learning focuses on incrementally training a model on a sequence of tasks with the aim of learning new tasks while minimizing performance drop on previous tasks. Existing approaches at the intersection of Continual Learning and Visual Question Answering (VQA) do not study how the multimodal nature of the input affects the learning dynamics of a model. In this paper, we demonstrate that each modality evolves at different rates across a continuum of tasks and that this behavior occurs in established encoder-only models as well as modern recipes for developing Vision {\&} Language (VL) models. Motivated by this observation, we propose a modality-aware feature distillation (MAFED) approach which outperforms existing baselines across models of varying scale in three multimodal continual learning settings. Furthermore, we provide ablations showcasing that modality-aware distillation complements experience replay. Overall, our results emphasize the importance of addressing modality-specific dynamics to prevent forgetting in multimodal continual learning.",
}
```
