# Enhancing Continual Learning in Visual Question Answering with Modality-Aware Feature Distillation
Code for Modality-Aware Feature Distillation ([MAFED](https://arxiv.org/abs/2406.19297)).

## Requirements
Install a conda environment:
We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards.
We use mixed-precision training hence GPUs with Tensor Cores are recommended.

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

To cite our paper:

```
@article{nikandrou2024enhancing,
  title={Enhancing Continual Learning in Visual Question Answering with Modality-Aware Feature Distillation},
  author={Nikandrou, Malvina and Pantazopoulos, Georgios and Konstas, Ioannis and Suglia, Alessandro},
  journal={arXiv preprint arXiv:2406.19297},
  year={2024}
}
```
