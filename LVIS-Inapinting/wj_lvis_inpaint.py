import argparse
import torch
import wandb
wandb.login()

import matplotlib.pyplot as plt
import numpy as np

from dataset import LVISDataset
from diffusers import StableDiffusionInpaintPipeline
from torch.utils.data import DataLoader


def my_collate_fn(batch):
    img_list, boxes_list, masks_list, areas_list, cats_list = [], [], [], [], []
    for sample in batch:
        img_list.append(sample[0])
        boxes_list.append(sample[1])
        masks_list.append(sample[2])
        areas_list.append(sample[3])
        cats_list.append(sample[4])
    return img_list, boxes_list, masks_list, areas_list, cats_list


def show_mask(mask, ax):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    x0, y0, w, h = box[0], box[1], box[2], box[3]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def main():
    # >>> load dataset >>>
    dataset = LVISDataset(data_root=args.data_root,
                          ann=args.ann)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            collate_fn=my_collate_fn,
                            shuffle=False)

    # >>> load model >>>
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
        cache_dir=args.model_checkpoint
    ).to("cuda")

    # >>> Inference >>>
    for cur_idx, (img_list, boxes_list, masks_list, areas_list, cats_list) in enumerate(dataloader):
        i = 0
        img, boxes, masks, areas, cats = img_list[i], boxes_list[i],masks_list[i], areas_list[i], cats_list[i]
        cats = [cat.replace('_', ' ') for cat in cats]

        from IPython import embed
        embed()

        w, h = img.size
        # img-boxs-mask
        plt.figure(figsize=(w / 60, h / 60))
        ax1 = plt.gca()
        ax1.axis('off')
        ax1.imshow(img)
        if len(boxes) > 0:
            for (box, label) in zip(boxes, cats):
                show_box(box, ax1, label)
            for mask in masks:
                show_mask(mask, ax1)
        fig_img_box_mask = plt.gcf()

        # mask only
        plt.figure(figsize=(w / 60, h / 60))
        ax2 = plt.gca()
        ax2.axis('off')
        ax2.imshow(img)
        if len(boxes) > 0:
            for mask in masks:
                show_mask(mask, ax2)
        fig_mask = plt.gcf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--ann', type=str)
    parser.add_argument('--model_checkpoint', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    run = wandb.init(project='LVIS-Inpaint')
    main()
    wandb.finish()
