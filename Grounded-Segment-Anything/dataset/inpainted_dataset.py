import os
import json

from torch.utils.data import Dataset
from PIL import Image


class InpaintedDataset(Dataset):
    def __init__(self, data_root, ann):
        with open(ann, 'r') as f:
            ann = json.load(f)
        self.data_root = data_root
        self.file_list = ann['selected_directory']

    def __len__(self):
        return self.file_list

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        path = os.path.join(self.data_root, filename)
        image = Image.open(os.path.join(path, 'image.jpg')).convert('RGB')
        inpainted_image = Image.open(os.path.join(path, 'inpainted_image.jpg')).convert('RGB')
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        return image, inpainted_image, metadata
