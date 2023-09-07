import argparse
import os
import json
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as TS

# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor, build_sam_hq, build_sam_hq_vit_l
from segment_anything.utils.transforms import ResizeLongestSide

# Tag2Text
import sys
sys.path.append('Tag2Text')
from Tag2Text.models import tag2text
from Tag2Text import inference_tag2text

# BLIP-2
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# wandb
import wandb
wandb.login()
run = wandb.init('Tag2Text & Grounded DINO & HQ-SAM & Visual Genome')

from datasets import load_dataset
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from torch.utils.data import DataLoader


def my_collate_fn(batch):
    images, Ws, Hs, paths = [], [], [], []
    for item in batch:
        images.append(item[0])
        Ws.append(item[1])
        Hs.append(item[2])
        paths.append(item[3])
    return [images, Ws, Hs, paths]


def load_grounding_dino_model(model_config_path, model_checkpoint_path):
    """load groundingdino model"""
    cfg = SLConfig.fromfile(model_config_path)
    cfg.device = "cuda"
    model = build_model(cfg)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model.cuda()


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
    # preprocess captions
    for k in range(len(caption)):
        caption[k] = caption[k].lower().strip()
        if not caption[k].endswith("."):
            caption[k] = caption[k] + "."
    # forward grounded dino
    with torch.no_grad():
        outputs = model(image, captions=caption)
    logits = outputs["pred_logits"].cpu().sigmoid()  # (bs, nq, 256)
    boxes = outputs["pred_boxes"].cpu()  # (bs, nq, 4)
    # post process
    boxes_list, scores_list, phrases_list = [], [], []
    for ub_logits, ub_boxex, cap in zip(logits, boxes, caption):
        mask = ub_logits.max(dim=1)[0] > box_threshold
        logits_filtered = ub_logits[mask]  # (n, 256)
        boxes_filtered = ub_boxex[mask]  # (n, 4)
        phrases_filtered = []
        scores_filtered = []
        for logit, box in zip(logits_filtered, boxes_filtered):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, model.tokenizer(cap), model.tokenizer)
            if with_logits:
                phrases_filtered.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                phrases_filtered.append(pred_phrase)
            scores_filtered.append(logit.max().item())
        boxes_list.append(boxes_filtered)
        scores_list.append(torch.Tensor(scores_filtered))
        phrases_list.append(phrases_filtered)
    return boxes_list, scores_list, phrases_list


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
            'original_size': (Hs[i], Ws[i])
        }
        if boxes[i].shape[0] > 0:
            data['boxes'] = resize_transform.apply_boxes_torch(boxes[i], (Hs[i], Ws[i]))
        batched_input.append(data)
    return batched_input


def wandb_visualize(images, tags, captions, boxes_filt, masks_list, pred_phrases):
    for i in range(len(images)):
        img, tag, caption, boxes, masks, labels = images[i], tags[i], captions[i], boxes_filt[i], masks_list[i], pred_phrases[i]
        if len(boxes) > 0:
            fig, ax = plt.subplots(1, 3, figsize=(10, 10))
            # show image only
            ax[0].imshow(img)
            ax[0].axis('off')
            # show boxes, masks, image
            ax[1].imshow(img)
            for (box, label) in zip(boxes, labels):
                show_box(box.cpu().numpy(), ax[1], label)
            for mask in masks:
                show_mask(mask, ax[1], random_color=True)
            # mask_all = np.logical_or.reduce(masks, axis=0)
            # show_mask(mask_all, ax[1], random_color=True)
            ax[1].axis('off')
            # show masks only
            for mask in masks:
                show_mask(mask, ax[2], random_color=True)
            ax[2].axis('off')
            # send to wandb
            plt.tight_layout()
            run.log({'Visualization': wandb.Image(plt.gcf(), caption=f'Tags: {tag}\nCaption:{caption}')})
            plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--tag2text_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_checkpoint", type=str, help="path to checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, help="path to checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--data_root", type=str)
    parser.add_argument("--data_ann", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--user_specified_tags", type=str, default="None",
                        help="user specified tags for tag2text, if more than one, use ',' to split")
    parser.add_argument("--grounding_dino_img_size", type=int, default=800)

    args = parser.parse_args()

    # cfg
    config_file = args.config
    device = "cuda"

    # load data
    # dataset = CoCoDataset(image_root=args.data_root, json=args.data_ann)
    dataset = load_dataset(path="visual_genome", name="objects_v1.2.0", split="train")
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=my_collate_fn)

    # load Grounded-DINO model
    grounding_dino_model = load_grounding_dino_model(config_file, args.grounded_checkpoint)

    # load hq-SAM
    if args.use_sam_hq:
        sam = build_sam_hq_vit_l(checkpoint=args.sam_hq_checkpoint).cuda()
    else:
        sam = build_sam(checkpoint=args.sam_checkpoint).cuda()

    # load Tag2Text model
    # filter out attributes and action categories which are difficult to grounding
    delete_tag_index = []
    for i in range(3012, 3429):
        delete_tag_index.append(i)
    tag2text_model = tag2text.tag2text_caption(pretrained=args.tag2text_checkpoint,
                                               delete_tag_index=delete_tag_index,
                                               image_size=384,
                                               vit='swin_b').cuda()
    tag2text_model.threshold = 0.64  # we reduce the threshold to obtain more tags
    tag2text_model.eval()

    # iterate forward pass
    total_iter = len(dataloader)
    result_dict = {'configure': vars(args)}
    for iter_idx, (images, Ws, Hs, paths) in enumerate(dataloader):

        # >>> Tagging: inference tag2text >>>
        trans_tag2text = TS.Compose([TS.Resize((384, 384)),
                                     TS.ToTensor(),
                                     TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        images_tag2text = torch.stack([trans_tag2text(img) for img in images]).cuda()
        tag2text_ret = inference_tag2text.inference(image=images_tag2text,
                                                    model=tag2text_model,
                                                    input_tag=args.user_specified_tags)
        tags_list = [tag.replace(' |', ',') for tag in tag2text_ret[0]]
        tag2text_captions_list = tag2text_ret[2]
        # empty cache
        torch.cuda.empty_cache()

        # >>> Detection: inference grounded dino >>>
        # preprocess images
        trans_grounded = TS.Compose(
            [
                TS.Resize((args.grounding_dino_img_size, args.grounding_dino_img_size)),
                TS.ToTensor(),
                TS.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        dino_images = torch.stack([trans_grounded(img) for img in images], dim=0).cuda()
        # forward grounded dino
        boxes_filt_list, scores_list, pred_phrases_list = get_grounding_output(
            model=grounding_dino_model,
            image=dino_images,
            caption=tags_list,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold
        )
        # post process bounding box
        for i in range(len(boxes_filt_list)):
            H, W = Hs[i], Ws[i]
            boxes = boxes_filt_list[i]
            for k in range(boxes.size(0)):
                boxes[k] = boxes[k] * torch.Tensor([W, H, W, H])
                boxes[k][:2] -= boxes[k][2:] / 2
                boxes[k][2:] += boxes[k][:2]
            boxes_filt_list[i] = boxes.cuda()
        # use NMS to handle overlapped boxes
        for i in range(args.batch_size):
            boxes_filt_list[i] = boxes_filt_list[i].cpu()
            nms_idx = torchvision.ops.nms(boxes_filt_list[i], scores_list[i], args.iou_threshold).numpy().tolist()
            boxes_filt_list[i] = boxes_filt_list[i][nms_idx].cuda()
            pred_phrases_list[i] = [pred_phrases_list[i][idx] for idx in nms_idx]
            # caption = check_caption(tag2text_caption, pred_phrases)
        # empty cache
        torch.cuda.empty_cache()

        # >>> Segmentation: inference sam >>>
        # preprocess images
        batched_input = prepare_sam_data(images=images, boxes=boxes_filt_list,
                                         Hs=Hs, Ws=Ws,
                                         resize_size=sam.image_encoder.img_size)
        # forward sam
        batched_output = sam(batched_input, multimask_output=False)
        masks_list = [output['masks'].cpu().numpy() for output in batched_output]
        # empty cache
        torch.cuda.empty_cache()

        # >>> Caption: BLIP-2
        # processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        # model = Blip2ForConditionalGeneration.from_pretrained(
        #     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32
        # ).cuda()
        # prompt = "Question: how many cats are there? Answer:"
        # inputs = processor(images=images,  return_tensors="pt").to(device, torch.float16)
        # generated_ids = model.generate(**inputs)
        # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # >>> Wandb visualize >>>
        wandb_visualize(images, tags_list, tag2text_captions_list, boxes_filt_list, masks_list, pred_phrases_list)

        # >>> Inpainting: inference stable diffusion or lama >>>

        from IPython import embed
        embed()
