import os
import shutil

from huggingface_hub import hf_hub_download

REPOS_AND_FILES = {
    "gpantaz/vl-pythia-eva-160m": [
        "checkpoint-final/model.safetensors",
        "checkpoint-final/config.json",
        "checkpoint-final/generation_config.json",
    ],
    "gpantaz/vl-pythia-eva-410m": [
        "checkpoint-final/model.safetensors",
        "checkpoint-final/config.json",
        "checkpoint-final/generation_config.json",
    ],
    "gpantaz/vl-pythia-eva-1b": [
        "checkpoint-final/model-00001-of-00002.safetensors",
        "checkpoint-final/model-00002-of-00002.safetensors",
        "checkpoint-final/model.safetensors.index.json",
        "checkpoint-final/config.json",
        "checkpoint-final/generation_config.json",
    ],
}
DOWNLOAD_DIR = "storage/download_dir"
TARGET_DIR = "storage/models"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)

for repo, files in REPOS_AND_FILES.items():
    local_dir = os.path.join(TARGET_DIR, os.path.basename(repo))
    if os.path.exists(local_dir):
        print(f"Skipping {repo} as it already exists.")
        continue
    cache_dir = os.path.join(DOWNLOAD_DIR, ".cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    print(f"Downloading {repo}...")
    for file in files:
        hf_hub_download(repo_id=repo, filename=file, local_dir=DOWNLOAD_DIR)
    os.rename(os.path.join(DOWNLOAD_DIR, "checkpoint-final"), local_dir)

shutil.rmtree(DOWNLOAD_DIR)
