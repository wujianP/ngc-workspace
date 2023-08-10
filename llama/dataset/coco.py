from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, image_root, json):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            image_root: image directory.
            json: coco annotation file path.
            transforms: image transformer.
        """
        self.root = image_root
        self.dataset = COCO(json)
        self.ids = list(self.dataset.anns.keys())

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        dataset = self.dataset
        ann_id = self.ids[index]
        caption = dataset.anns[ann_id]['caption'].strip()
        # img_id = dataset.anns[ann_id]['image_id']
        # path = dataset.loadImgs(img_id)[0]['file_name']
        # image = Image.open(os.path.join(self.root, path)).convert('RGB')

        # return ann_id, caption
        return caption

    def __len__(self):
        return len(self.ids)
