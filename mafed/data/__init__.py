from mafed.data.base_data import BaseDataset
from mafed.data.loader import PrefetchLoader
from mafed.data.vl_pythia_vqa_dataset import VLPythiaVQADataset, vlpythia_vqa_collate

datasets_map = {
    "train": {"vlpythia": VLPythiaVQADataset},
    "valid": {"vlpythia": VLPythiaVQADataset},
}

collate_fn = {
    "train": {"vlpythia": vlpythia_vqa_collate},
    "valid": {"vlpythia": vlpythia_vqa_collate},
}
