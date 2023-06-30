import torch
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    def __init__(self, caption_path, prompt_template):
        self.caption_path = caption_path
        self.prompt_template = prompt_template
        self.cap_dict = torch.load(caption_path)
        self.filenames = list(self.cap_dict.keys())

    def __len__(self):
        return len(self.cap_dict)

    def __getitem__(self, index):
        filename = self.filenames[index]
        caption = str(self.cap_dict[filename]).replace('\'', '')
        action = filename.split('/')[-2].replace('_', ' ')
        prompt = self.prompt_template + caption + f', {action}, \nOutput:'
        return filename, prompt, caption, action
