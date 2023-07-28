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
from matplotlib import pyplot as plt


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
    for cur_iter, (images, heights, widths, paths) in enumerate(dataloader):
        start_time = time.time()

        # forward pass
        images = images.cuda()
        outputs = model(images)
        logits = outputs.logits

        for i in range(args.batch_size):
            # rescale logits to original image size
            logit = torch.unsqueeze(logits[i], 0)
            logit = nn.functional.interpolate(logit, size=(heights[i], widths[i]), mode='bilinear', align_corners=False)
            # get segment mask labels
            seg_mask = logit.argmax(dim=1)[0]

            from IPython import embed
            embed()

            # color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
            # palette = np.array(ade_palette())
            # for label, color in enumerate(palette):
            #     color_seg[seg == label, :] = color
            # # Convert to BGR
            # color_seg = color_seg[..., ::-1]
            #
            # # Show image + mask
            # img = np.array(image) * 0.5 + color_seg * 0.5
            # img = img.astype(np.uint8)
            #
            # plt.figure(figsize=(15, 10))
            # plt.imshow(img)
            # plt.show()

        batch_time = time.time() - start_time

        print(f'[ITER: {cur_iter+1} / [{total_iters}], BATCH TIME: {batch_time:.3f}')


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
