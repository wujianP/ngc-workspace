"""
This is a script that used to segment images using Grounded-Segment-Anything
"""
import argparse
import os
import json
import torch
import time

# Grounding DINO
import groundingdino.datasets.transforms as T
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
from segment_anything.utils.transforms import ResizeLongestSide

# dataset
from dataset import RawFrameDatasetGroundingSAM

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

# wandb
import wandb

wandb.login()

from utils import my_collate_fn


def load_grounding_dino_model(model_config_path, model_checkpoint_path):
    """load groundingdino model"""
    args = SLConfig.fromfile(model_config_path)
    args.device = 'cuda'
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model.cuda()


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
    # preprocess caption (text prompt)
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    # forward
    with torch.no_grad():
        outputs = model(image, captions=[caption] * args.batch_size)
    logits = outputs["pred_logits"].cpu().sigmoid()  # (bs, nq, 256)
    boxes = outputs["pred_boxes"].cpu()  # (bs, nq, 4)
    # post process
    boxes_list, phrases_list = [], []
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    for ub_logits, ub_boxex in zip(logits, boxes):
        mask = ub_logits.max(dim=1)[0] > box_threshold
        logits_filtered = ub_logits[mask]  # (n, 256)
        boxes_filtered = ub_boxex[mask]  # (n, 4)
        phrases_filtered = []
        for logit, box in zip(logits_filtered, boxes_filtered):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                phrases_filtered.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                phrases_filtered.append(pred_phrase)
        boxes_list.append(boxes_filtered)
        phrases_list.append(phrases_filtered)

    return boxes_list, phrases_list


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


def prepare_grounding_dino_data(images):
    """images is a list, each element is a PIL image object"""
    trans = T.Compose(
        [
            T.Resize((args.grounding_dino_img_size, args.grounding_dino_img_size)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dino_images = [trans(image).cuda() for image in images]
    dino_images = torch.stack(dino_images, dim=0)
    return dino_images


def prepare_sam_data(images, boxes, Hs, Ws, resize_size):
    resize_transform = ResizeLongestSide(resize_size)

    def prepare_image(image, transform):
        image = np.array(image)
        image = transform.apply_image(image)
        image = torch.as_tensor(image).cuda()
        return image.permute(2, 0, 1).contiguous()

    batched_input = []
    for i in range(len(images)):
        data = {
            'image': prepare_image(images[i], resize_transform),
            'boxes': resize_transform.apply_boxes_torch(boxes[i], (Hs[i], Ws[i])),
            'original_size': (Hs[i], Ws[i])
        }
        batched_input.append(data)
    return batched_input


@torch.no_grad()
def main(agrs):
    # cfg
    config_file = args.config

    # make dir
    os.makedirs(args.output_dir, exist_ok=True)

    # load dataset
    dataset = RawFrameDatasetGroundingSAM(path_file=agrs.data_path)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=agrs.batch_size,
                            num_workers=agrs.num_workers,
                            shuffle=False,
                            collate_fn=my_collate_fn,
                            pin_memory=True)

    # initialize grounding-dino model
    grounding_dino_model = load_grounding_dino_model(config_file,
                                                     agrs.grounded_checkpoint)

    # initialize segment anything model
    if args.use_sam_hq:
        sam = build_sam_hq_vit_b(checkpoint=agrs.sam_hq_checkpoint).cuda()
    else:
        sam = build_sam(checkpoint=agrs.sam_checkpoint).cuda()

    # iterate forward pass
    total_iter = len(dataloader)
    for iter_idx, (images, Ws, Hs, paths) in enumerate(dataloader):
        start_time = time.time()

        # transform image for dino
        dino_images = prepare_grounding_dino_data(images)

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model=grounding_dino_model,
            image=dino_images,
            caption=agrs.text_prompt,
            box_threshold=agrs.box_threshold,
            text_threshold=agrs.text_threshold
        )

        # post process bounding box
        for i in range(len(boxes_filt)):
            H, W = Hs[i], Ws[i]
            boxes = boxes_filt[i]
            for k in range(boxes.size(0)):
                boxes[k] = boxes[k] * torch.Tensor([W, H, W, H])
                boxes[k][:2] -= boxes[k][2:] / 2
                boxes[k][2:] += boxes[k][:2]
            boxes_filt[i] = boxes.cuda()

        torch.cuda.empty_cache()
        del dino_images

        ground_dino_time = time.time() - start_time

        # prepare data for sam input
        resize_size = agrs.sam_img_size if agrs.sam_img_size else sam.image_encoder.image_size
        batched_input = prepare_sam_data(images=images, boxes=boxes_filt,
                                         Hs=Hs, Ws=Ws,
                                         resize_size=resize_size)

        # run sam model
        batched_output = sam(batched_input, multimask_output=False)
        masks_list = [output['masks'].cpu().numpy() for output in batched_output]

        sam_time = time.time() - start_time - ground_dino_time
        batch_time = time.time() - start_time

        print(f'BATCH: [{iter_idx + 1} / {total_iter}], TIME: [batch-{batch_time:.3f} dino-{ground_dino_time:.3f} sam-{sam_time:.3f}]')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything", add_help=True)
    parser.add_argument('--data_path', type=str, required=True, help='path to the data annotation file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--grounding_dino_img_size', type=int, default=512)
    parser.add_argument('--sam_img_size', type=int, default=None)

    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cuda", help="running on cuda")
    args = parser.parse_args()

    main(args)
