import torch
import json
import os
import ast

import pandas as pd

from PIL import Image
from pycocotools.coco import COCO
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


class CocoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, image_root, ann):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            image_root: image directory.
            json: coco annotation file path.
            transforms: image transformer.
        """
        self.root = image_root
        self.dataset = COCO(ann)
        self.ids = list(self.dataset.anns.keys())

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        dataset = self.dataset
        ann_id = self.ids[index]
        caption = dataset.anns[ann_id]['caption'].strip()
        img_id = dataset.anns[ann_id]['image_id']
        path = dataset.loadImgs(img_id)[0]['file_name']
        #
        # image = Image.open(os.path.join(self.root, path)).convert('RGB')
        # if self.transforms is not None:
        #     image = self.transforms(image)

        return caption, ann_id, img_id, path

    def __len__(self):
        return len(self.ids)


class NegCLIPCocoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, image_root, ann):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            image_root: image directory.
            ann: coco annotation file path.
        """
        self.root = image_root
        print(f'loading data from {ann}')
        df = pd.read_csv(ann, sep="\t", converters={"neg_caption": ast.literal_eval, "neg_image": ast.literal_eval})
        self.image_paths = df["filepath"].tolist()
        self.captions = df["title"].tolist()
        self.hard_captions = df["neg_caption"].tolist()
        self.hard_image_ids = df["neg_image"].tolist()
        print('Done loading data.')

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        caption = self.captions[index]
        image_path = self.image_paths[index]
        hard_image_path = [self.image_paths[hard_img_idx] for hard_img_idx in self.hard_image_ids[index]]

        return caption, image_path, hard_image_path

    def __len__(self):
        return len(self.captions)
