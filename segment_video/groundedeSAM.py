"""
This is a script that used to segment images using Grounded-Segment-Anything
"""
import argparse
import os
import json
import torch
import time

# Grounding DINO
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from torch.utils.data import DataLoader

# segment anything
from segment_anything import (
    build_sam,
    build_sam_hq_vit_b,
    SamPredictor
)

# dataset
from dataset import RawFrameDatasetGroundingSAM

import cv2
import numpy as np
import matplotlib.pyplot as plt

# wandb
import wandb
wandb.login()


def load_model(model_config_path, model_checkpoint_path):
    """load groundingdino model"""
    args = SLConfig.fromfile(model_config_path)
    args.device = 'cuda'
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    image = image.to('cuda')
    with torch.no_grad():
        outputs = model(image, captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


@torch.no_grad()
def main(agrs):

    from IPython import embed
    embed()

    # cfg
    config_file = args.config

    # make dir
    os.makedirs(args.output_dir, exist_ok=True)

    # load dataset
    dataset = RawFrameDatasetGroundingSAM(path_file=agrs.data_path_file)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=agrs.batch_size,
                            num_workers=agrs.num_workers,
                            shuffle=False,
                            pin_memory=True)

    # initialize grounding-dino model
    grounding_dino_model = load_model(config_file, agrs.grounded_checkpoint).cuda()

    # initialize segment anything model
    if args.use_sam_hq:
        predictor = SamPredictor(build_sam_hq_vit_b(checkpoint=agrs.sam_hq_checkpoint).to('cuda'))
    else:
        predictor = SamPredictor(build_sam(checkpoint=agrs.sam_checkpoint).to('cuda'))

    # iterate forward pass
    start_time = time.time()
    for iter_idx, (images, Ws, Hs, paths) in enumerate(dataloader):
        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model=grounding_dino_model,
            image=images,
            caption=agrs.text_prompt,
            box_threshold=agrs.box_threshold,
            text_threshold=agrs.text_threshold
        )

        # image should be ndarray
        predictor.set_image(images)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to('cuda')

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to('cuda'),
            multimask_output=False,
        )

        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)

        plt.axis('off')
        plt.savefig(
            os.path.join(args.output_dir, "grounded_sam_output.jpg"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )

        save_mask_data(agrs.output_dir, masks, boxes_filt, pred_phrases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything", add_help=True)
    parser.add_argument('--data_path_file', type=str, required=True, help='path to the data annotation file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")
    parser.add_argument("--image_path", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cuda", help="running on cuda")
    args = parser.parse_args()

    main(args)

