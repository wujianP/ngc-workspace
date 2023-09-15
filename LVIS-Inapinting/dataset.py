import os

from PIL import Image
from torch.utils.data import Dataset
from lvis import LVIS


class LVISDataset(Dataset):
    def __init__(self, data_root, ann):
        self.data_root = data_root
        self.lvis = LVIS(ann)
        self.image_ids = self.lvis.get_img_ids()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # load image
        img_id = self.image_ids[idx]
        img_dict = self.lvis.load_imgs([img_id])[0]
        img_filename = '/'.join(img_dict['coco_url'].split('/')[-2:])
        img_path = os.path.join(self.data_root, img_filename)
        img = Image.open(img_path).convert('RGB')

        # load masks
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        ann_dicts = self.lvis.load_anns(ann_ids)
        boxes, masks, areas, cats = [], [], [], []
        for ann_dict in ann_dicts:
            mask = self.lvis.ann_to_mask(ann_dict)
            box = ann_dict['bbox']  # [x,y,w,h]
            area = ann_dict['area']
            cat_id = ann_dict['category_id']
            cat = self.lvis.load_cats([cat_id])[0]['name']
            boxes.append(box)
            masks.append(mask)
            areas.append(area)
            cats.append(cat)
        return img, boxes, masks, areas, cats
