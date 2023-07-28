import torch
import time
import wandb

import numpy as np

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torch import nn
from dataset import RawFrameDataset
from utils import ade_palette
from torchvision.transforms import ToPILImage


@torch.no_grad()
def main(args):
    # init model
    feature_extractor = SegformerFeatureExtractor.from_pretrained(args.model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_path).cuda()

    # init data
    dataset = RawFrameDataset(path_file=args.data_path_file, feature_extractor=feature_extractor)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False)

    # segment forward pass
    total_iters = len(dataloader)
    for cur_iter, (images, paths) in enumerate(dataloader):
        start_time = time.time()

        from IPython import embed
        embed()

        # forward pass
        images_pt = feature_extractor(images, return_tensors="pt").pixel_values[0].cuda()
        outputs = model(images_pt)
        logits = outputs.logits

        for i in range(args.batch_size):
            # image
            img = images[i]
            height, width, _ = img.shape
            # rescale logits to original image size
            logit = torch.unsqueeze(logits[i], 0)
            logit = nn.functional.interpolate(logit, size=(height, width), mode='bilinear', align_corners=False)
            # get segment mask labels
            seg_mask = logit.argmax(dim=1)[0].cpu()
            # plot to wandb
            if i == 0:
                visualize_result(seg_mask=seg_mask, image_raw=images[i], stepIdx=cur_iter+1)

        batch_time = time.time() - start_time

        print(f'[ITER: {cur_iter+1} / [{total_iters}], BATCH TIME: {batch_time:.3f}')


def visualize_result(seg_mask, image_raw, stepIdx):
    # get segment color map
    seg_color_map = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        seg_color_map[seg_mask == label, :] = color
    seg_color_map = seg_color_map[..., ::-1]  # Convert to BGR
    # get image
    image_raw = image_raw.numpy()
    # get mixed map
    mix = image_raw * 0.4 + seg_color_map * 0.6
    # wandb
    wandb.log({"segment-visualize": [wandb.Image(image_raw), wandb.Image(seg_color_map), wandb.Image(mix)]}, step=stepIdx)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='nvidia/segformer-b0-finetuned-ade-512-512')
    parser.add_argument('--data_path_file', type=str, default='/discobox/wjpeng/dataset/k400/ann/rawframe_list.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    wandb_logger = wandb.init('Segformer on Kinetics400')
    main(args)
    wandb.finish()
