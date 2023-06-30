import torch
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    def __init__(self, caption_path):
        self.caption_path = caption_path
        self.cap_dict = torch.load(caption_path)
        self.filenames = list(self.cap_dict.keys())

    def __len__(self):
        return len(self.cap_dict)

    def __getitem__(self, index):
        filename = self.filenames[index]
        caption = self.cap_dict[filename]
        caption = str(caption).replace('\'', '')
        action = filename.split('/')[-2].replace('_', ' ')
        return filename, caption, action
