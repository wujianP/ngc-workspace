import argparse
import torch
import wandb
wandb.login()

from datasets import load_dataset
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


def main():
    # load dataset
    dataset = load_dataset("winvoker/lvis", cache_dir=args.data_root, split='validation')

    # load Stable-Diffusion-Inpaint
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
        cache_dir=args.model_checkpoint
    ).to("cuda")

    from IPython import embed
    embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--model_checkpoint', type=str)
    args = parser.parse_args()

    run = wandb.init('LVIS-Inpaint')
    main()
    wandb.finish()
