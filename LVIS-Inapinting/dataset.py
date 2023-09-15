import json

from PIL import Image
from torch.utils.data import Dataset
from lvis import LVIS


class LVISDataset(Dataset):
    def __init__(self, data_root, ann):
        self.data_root = data_root
        ann = LVIS(ann)
        self.images = ann.imgs
        from IPython import embed
        embed()
