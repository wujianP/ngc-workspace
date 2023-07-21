import torch
import json
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    """dataset for Kinetics400 frame-level captions"""
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


class SMiTCaptionDataset(Dataset):
    """dataset for S-MiT original captions"""
    def __init__(self, caption_path, prompt_template):
        assert caption_path.endswith('.json')
        self.caption_path = caption_path
        self.prompt_template = prompt_template
        with open(caption_path, 'r') as file:
            self.data_list = json.load(file)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        file_path = data['file_path']
        caption = data['caption']
        class_name = data['class_name']
        file_name = data['file_name']
        prompt = self.prompt_template + caption + ', Output:'

        return file_path, prompt, caption, class_name, file_name
