import argparse
import torch
import wandb
wandb.login()

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
        from IPython import embed
        embed()


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
