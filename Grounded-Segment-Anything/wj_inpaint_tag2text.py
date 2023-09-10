import argparse
import torch
import torchvision
import random
import time
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

# wandb
import wandb

wandb.login()

from datasets import load_dataset
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader


def my_collate_fn(batch):
    images, image_ids, Ws, Hs, objects = [], [], [], [], []
    for item in batch:
        images.append(item['image'])
        image_ids.append(item['image_id'])
        Ws.append(item['width'])
        Hs.append(item['height'])
        objects.append(item['objects'])
    return [images, image_ids, Ws, Hs, objects]


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
            # wj: only keep the most confident one
            posmap = (logit > text_threshold) * (logit == logit.max())
            pred_phrase = get_phrases_from_posmap(posmap, model.tokenizer(cap), model.tokenizer)
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


def show_box(box, ax, label, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2))
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


def wandb_visualize(images, tags, captions, boxes_filt, masks_list, pred_phrases, inpaint_masks, after_inpaint_images, inpainted_tag_lis):
    for i in range(len(images)):
        img, tag, caption, boxes, masks, labels, ipt_img, ipt_mask, ipt_tag = images[i], tags[i], captions[i],\
            boxes_filt[i], masks_list[i], pred_phrases[i], after_inpaint_images[i], inpaint_masks[i], inpainted_tag_lis[i]
        w, h = img.size
        # masks, boxes and image
        plt.figure(figsize=(w/80, h/80))
        ax1 = plt.gca()
        ax1.axis('off')
        ax1.imshow(img)
        if len(boxes) > 0:
            for (box, label) in zip(boxes, labels):
                show_box(box.cpu().numpy(), ax1, label)
            for mask in masks:
                show_mask(mask, ax1, random_color=True)
        fig1 = plt.gcf()
        # all mask only
        plt.figure(figsize=(w/100, h/100))
        ax2 = plt.gca()
        ax2.axis('off')
        if len(boxes) > 0:
            for mask in masks:
                show_mask(mask, ax2, random_color=True)
        fig2 = plt.gcf()

        # show
        run.log({'inpaint': [wandb.Image(img, caption=f'raw image\ntag2text caption: {caption}'),
                             wandb.Image(ipt_img, caption='inpainted image'),
                             wandb.Image(ipt_mask, caption=f'inpainted mask\ninpaint tags: {ipt_tag}'),
                             wandb.Image(fig1, caption=f'masks and boxes\ntag2text tags: {tag}'),
                             wandb.Image(fig2, caption=f'masks')]})


def filter_and_select_bounding_boxes_and_masks(bounding_boxes, masks, tags, W, H, n,
                                               high_threshold, low_threshold, mask_threshold, tag2cluster):
    """
    1. Filter out boxes that are too large or too small in area.
    2. Filter out boxes where the proportion of the mask to the box is too small.
    3. Merge the masks that have not been filtered out according to their respective categories.
    4. Select one mask to return.
    5. Special case: If there are no masks that meet the criteria, then randomly generate a rectangular mask and return a flag.
    :param tag2cluster:
    :param bounding_boxes:
    :param masks:
    :param tags:
    :param W:
    :param H:
    :param n:
    :param high_threshold:
    :param low_threshold:
    :param mask_threshold:
    :return: [selected mask, ...], [selected tag, ...], flag(Bool)
    """
    # generate mock mask map when no valid mask to be selected
    mock_mask = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(mock_mask)
    draw.rectangle([(W // 4, H // 4), (3 * W // 4, 3 * H // 4)], fill=255)
    mock_mask = np.array(mock_mask) > 0  # to bool array

    if len(bounding_boxes) == 0:
        return [mock_mask], ['no-valid-mask'], True

    # preprocess tags format
    tags = np.array([tag.split('(')[0] for tag in tags])  # from 'cls1(0.27)' to 'cls'

    # First: filter all masks and boxes not statisfy requirements
    selected_idx = []
    for i, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = box.cpu()
        box_area = (x2 - x1) * (y2 - y1)
        box_percentage = box_area / (W * H)
        mask_area = masks[i].sum()
        mask_percentage = mask_area / box_area

        # filter bounding box with too large or small area
        if box_percentage >= high_threshold or box_percentage <= low_threshold:
            continue

        # filter masks with to small area
        if mask_percentage <= mask_threshold:
            continue
        # select
        selected_idx.append(i)

    if len(selected_idx) == 0:
        return [mock_mask], ['no-valid-mask'], True

    # Then: merge all masks with the same category
    # with tag clustering
    selected_masks = masks[selected_idx]
    selected_tags = tags[selected_idx]
    tag_cluster_ids = [tag2cluster[tag] for tag in selected_tags]
    unique_cluster_ids = np.unique(tag_cluster_ids)
    merged_masks = []
    merged_tags = []
    for cluster_ids in unique_cluster_ids:
        mask_indices = np.where(tag_cluster_ids == cluster_ids)
        masks_with_same_tag = selected_masks[mask_indices]
        merged_mask = np.logical_or.reduce(masks_with_same_tag, axis=0)[0]
        merged_tag = np.unique(selected_tags[mask_indices])
        merged_masks.append(merged_mask)
        merged_tags.append(merged_tag)

    # no tag clustering
    # merged_tags = np.unique(selected_tags)
    # merged_masks = []
    # for tag in merged_tags:
    #     mask_indices = np.where(selected_tags == tag)
    #     masks_with_same_tag = selected_masks[mask_indices]
    #     merged_mask = np.logical_or.reduce(masks_with_same_tag, axis=0)[0]
    #     merged_masks.append(merged_mask)

    if n > len(merged_masks):
        return merged_masks, merged_tags, False
    else:
        sampled_indices = random.sample(range(len(merged_masks)), n)
        sampled_masks = [merged_masks[idx] for idx in sampled_indices]
        sampled_tags = [merged_tags[idx] for idx in sampled_indices]

    return sampled_masks, sampled_tags, False


@torch.no_grad()
def main():
    # load data
    dataset = load_dataset(path="visual_genome", name="objects_v1.2.0",
                           split="train", cache_dir=args.data_root)
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
    tag2text_model.threshold = args.tag2text_threshold  # we reduce the threshold to obtain more tags
    tag2text_model.eval()

    # load Stable-Diffusion-Inpaint
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
        cache_dir=args.sd_inpaint_checkpoint
    ).to("cuda")

    # iterate forward pass
    total_iter = len(dataloader)
    start_time = time.time()
    for iter_idx, (images, image_ids, Ws, Hs, objects) in enumerate(dataloader):
        # >>> load data >>>
        data_time = time.time() - start_time
        start_time = time.time()

        # >>> Tagging: inference tag2text >>>
        trans_tag2text = TS.Compose([TS.Resize((384, 384)),
                                     TS.ToTensor(),
                                     TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        tag2text_images = torch.stack([trans_tag2text(img) for img in images]).cuda()
        tag2text_ret = inference_tag2text.inference(image=tag2text_images,
                                                    model=tag2text_model,
                                                    input_tag=args.user_specified_tags)
        tags_list = [tag.replace(' | ', '.') for tag in tag2text_ret[0]]
        tag2text_captions_list = tag2text_ret[2]
        # empty cache
        torch.cuda.empty_cache()
        # tagging time
        tag_time = time.time() - start_time
        start_time = time.time()

        # >>> Detection: inference grounded dino >>>
        # > preprocess images >
        trans_grounded = TS.Compose(
            [
                TS.Resize((args.grounding_dino_img_size, args.grounding_dino_img_size)),
                TS.ToTensor(),
                TS.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        dino_images = torch.stack([trans_grounded(img) for img in images], dim=0).cuda()
        # > forward grounded dino >
        boxes_filt_list, scores_list, pred_phrases_list = get_grounding_output(
            model=grounding_dino_model,
            image=dino_images,
            caption=tags_list,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold
        )

        # > post process bounding box >
        for i in range(len(boxes_filt_list)):
            H, W = Hs[i], Ws[i]
            boxes = boxes_filt_list[i]
            for k in range(boxes.size(0)):
                boxes[k] = boxes[k] * torch.Tensor([W, H, W, H])
                boxes[k][:2] -= boxes[k][2:] / 2
                boxes[k][2:] += boxes[k][:2]
            boxes_filt_list[i] = boxes.cuda()
        # > use NMS to handle overlapped boxes >
        for i in range(args.batch_size):
            boxes_filt_list[i] = boxes_filt_list[i].cpu()
            nms_idx = torchvision.ops.nms(boxes_filt_list[i], scores_list[i], args.iou_threshold).numpy().tolist()
            boxes_filt_list[i] = boxes_filt_list[i][nms_idx].cuda()
            pred_phrases_list[i] = [pred_phrases_list[i][idx] for idx in nms_idx]
            # caption = check_caption(tag2text_caption, pred_phrases)
        # empty cache
        torch.cuda.empty_cache()
        # detection time
        det_time = time.time() - start_time
        start_time = time.time()

        # >>> Segmentation: inference sam >>>
        # > preprocess images >
        batched_input = prepare_sam_data(images=images, boxes=boxes_filt_list,
                                         Hs=Hs, Ws=Ws,
                                         resize_size=sam.image_encoder.img_size)
        # > forward sam >
        batched_output = sam(batched_input, multimask_output=False)
        masks_list = [output['masks'].cpu().numpy() for output in batched_output]
        # empty cache
        torch.cuda.empty_cache()
        # segmentation time
        seg_time = time.time() - start_time
        start_time = time.time()

        # >>> Inpainting: inference stable diffusion or lama >>>
        # > preprocess images
        inpaint_images = [img.resize((512, 512)) for img in images]
        # > preprocess segmentation masks >
        inpaint_masks = []
        inpaint_mask_flags = []
        selected_tags_list = []
        tag2cluster = np.load(args.clustered_tags, allow_pickle=True).tolist()['tag2cluster']
        for masks, boxes, pred_phrases, W, H in zip(masks_list, boxes_filt_list, pred_phrases_list, Ws, Hs):
            # choose the object to be masked
            selected_masks, selected_tags, no_valid_flag = filter_and_select_bounding_boxes_and_masks(
                bounding_boxes=boxes,
                tags=pred_phrases,
                masks=masks, W=W, H=H,
                n=args.inpaint_object_num,
                high_threshold=args.inpaint_select_upperbound,
                low_threshold=args.inpaint_select_lowerbound,
                mask_threshold=args.inpaint_mask_threshold,
                tag2cluster=tag2cluster)
            # merge selected masks
            mask = np.logical_or.reduce(selected_masks, axis=0)  # merge all masks
            mask = mask.astype(np.uint8)  # from bool to int
            # mask = Image.fromarray(mask).resize((512, 512))  # transform to PIL Image
            # dilate and erode
            kernel = np.ones((args.mask_dilate_kernel_size, args.mask_dilate_kernel_size), np.uint8)
            edge_kernel = np.ones((args.mask_dilate_edge_size, args.mask_dilate_edge_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)    # dilate
            mask = cv2.erode(mask, kernel, iterations=1)    # erode
            mask = cv2.dilate(mask, edge_kernel, iterations=1)    # dilate edge
            # mask = mask.filter(ImageFilter.MaxFilter(size=args.mask_dilate_edge_size))  # dilate the mask edge
            mask = mask * 255
            mask = Image.fromarray(mask).resize((512, 512))
            inpaint_masks.append(mask)
            inpaint_mask_flags.append(no_valid_flag)
            selected_tags_list.append(selected_tags)

        after_inpaint_images = inpaint_pipe(image=inpaint_images, prompt=[''] * args.batch_size,
                                            mask_image=inpaint_masks).images

        # resize to original size
        after_inpaint_images = [after_ipt_img.resize((w, h)) for after_ipt_img, w, h in zip(after_inpaint_images, Ws, Hs)]
        inpaint_masks = [ipt_mask.resize((w, h)) for ipt_mask, w, h in zip(inpaint_masks, Ws, Hs)]
        # empty cache
        torch.cuda.empty_cache()
        # inpainting time
        ipt_time = time.time() - start_time
        start_time = time.time()

        # >>> Wandb visualize >>>
        if (iter_idx+1) % args.visualize_freq == 0:
            wandb_visualize(images, tags_list, tag2text_captions_list, boxes_filt_list, masks_list, pred_phrases_list,
                            inpaint_masks, after_inpaint_images, selected_tags_list)
        # empty cache
        torch.cuda.empty_cache()
        # visualization time
        vis_time = time.time() - start_time
        start_time = time.time()

        # >>> Output: print and save
        print(f'[{iter_idx + 1}/{total_iter}]({(iter_idx + 1) / total_iter * 100:.2f}%): '
              f'data: {data_time:.2f} tag: {tag_time:.2f} det: {det_time:.2f} '
              f'seg: {seg_time:.2f} inpaint: {ipt_time:.2f} wandb: {vis_time:.2f} '
              f'total: {data_time + tag_time + det_time + seg_time + ipt_time + vis_time:.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visual Genome Inpainting Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--tag2text_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_checkpoint", type=str, help="path to checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, help="path to checkpoint file")
    parser.add_argument("--sd_inpaint_checkpoint", type=str, help="path to checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
    parser.add_argument("--tag2text_threshold", type=float, default=0.64, help="the lower the more tags detected")

    parser.add_argument("--data_root", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--user_specified_tags", type=str, default="None",
                        help="user specified tags for tag2text, if more than one, use ',' to split")
    parser.add_argument("--grounding_dino_img_size", type=int, default=800)
    parser.add_argument("--mask_dilate_edge_size", type=int, default=11, help="mast be an odd")
    parser.add_argument("--mask_dilate_kernel_size", type=int, default=15, help="mast be an odd")
    parser.add_argument("--inpaint_object_num", type=int, default=1)
    parser.add_argument("--inpaint_select_upperbound", type=float, default=0.4)
    parser.add_argument("--inpaint_select_lowerbound", type=float, default=0.01)
    parser.add_argument("--inpaint_mask_threshold", type=float, default=0.2)
    parser.add_argument("--visualize_freq", type=int, default=5)
    parser.add_argument("--clustered_tags", type=str)

    args = parser.parse_args()

    # cfg
    config_file = args.config
    device = "cuda"

    run = wandb.init('Tag2Text & Grounded DINO & HQ-SAM & Visual Genome')
    main()
    wandb.finish()
