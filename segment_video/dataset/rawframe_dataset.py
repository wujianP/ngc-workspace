import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class RawFrameDataset(Dataset):
    def __init__(self, path_file, feature_extractor=None):
        with open(path_file, 'r') as file:
            self.path_list = file.readlines()
            self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        path = self.path_list[idx].strip()
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        f.close()
        image = np.array(image)

        return image, path

    def __len__(self):
        return len(self.path_list)
