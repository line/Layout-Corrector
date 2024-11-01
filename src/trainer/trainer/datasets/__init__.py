from .publaynet import PubLayNetDataset
from .rico import Rico25Dataset
from .crello_bbox import CrelloBboxDataset

_DATASETS = [
    Rico25Dataset,
    PubLayNetDataset,
    CrelloBboxDataset,
]
DATASETS = {d.name: d for d in _DATASETS}
