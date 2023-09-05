import os

from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CoCoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, image_root, json, tokenizer=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            image_root: image directory.
            json: coco annotation file path.
        """
        self.root = image_root
        self.dataset = COCO(json)
        self.ids = list(self.dataset.anns.keys())
        self.tokenize = tokenizer

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        dataset = self.dataset
        ann_id = self.ids[index]
        # caption = dataset.anns[ann_id]['caption'].strip()
        img_id = dataset.anns[ann_id]['image_id']
        path = dataset.loadImgs(img_id)[0]['file_name']

        image_path = os.path.join(self.root, path)
        image = Image.open(image_path).convert('RGB')
        W, H = image.size

        return image, W, H, image_path

    def __len__(self):
        return len(self.ids)
