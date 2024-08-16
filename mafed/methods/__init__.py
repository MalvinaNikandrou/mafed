from mafed.methods.base import CLStrategy, Naive
from mafed.methods.distillation import FeatureDistillation
from mafed.methods.ewc import EWC
from mafed.methods.replay import ER

CLMethod = {
    "naive": Naive,
    "ewc": EWC,
    "replay": ER,
    "featdistill": FeatureDistillation,
}
