import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# segment anything
from segment_anything import build_sam, SamPredictor, build_sam_hq, build_sam_hq_vit_l
from segment_anything.utils.transforms import ResizeLongestSide

# transformer
from datasets import load_dataset
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

# wandb
import wandb

wandb.login()

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


def save_mask_data(output_dir, caption, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = {
        'caption': caption,
        'mask': [{
            'value': value,
            'label': 'background'
        }]
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data['mask'].append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'label.json'), 'w') as f:
        json.dump(json_data, f)


def prepare_sam_data(images, objects_list, Hs, Ws, resize_size):
    resize_transform = ResizeLongestSide(resize_size)

    boxes_list, obj_name_list = [], []
    for objects in objects_list:
        # transform from x,y,w,h -> xyxy format
        boxes = [torch.tensor([obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']]).cuda() for obj in objects]
        boxes = torch.stack(boxes, dim=0)
        boxes_list.append(boxes)
        # object names
        obj_names = [obj['names'][0] for obj in objects]
        obj_name_list.append(obj_names)

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
        if boxes_list[i].shape[0] > 0:
            data['boxes'] = resize_transform.apply_boxes_torch(boxes_list[i], (Hs[i], Ws[i]))
        batched_input.append(data)
    return batched_input, boxes_list, obj_name_list


def wandb_visualize(images, boxes_filt, masks_list, pred_phrases):
    for i in range(len(images)):
        img, boxes, masks, labels = images[i], boxes_filt[i], masks_list[i], pred_phrases[i]
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
            run.log({'Visualization': wandb.Image(plt.gcf())})
            plt.close()


@torch.no_grad()
def main():
    # load data
    vg_obj = load_dataset(path="visual_genome", name="objects_v1.2.0", split="train")
    dataloader = DataLoader(dataset=vg_obj,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=my_collate_fn)

    # load hq-SAM
    if args.use_sam_hq:
        sam = build_sam_hq_vit_l(checkpoint=args.sam_hq_checkpoint).cuda()
    else:
        sam = build_sam(checkpoint=args.sam_checkpoint).cuda()

    # iterate forward pass
    total_iter = len(dataloader)
    result_dict = {'configure': vars(args)}
    for iter_idx, (images, image_ids, Ws, Hs, objects) in enumerate(dataloader):
        # >>> Segmentation: SAM-HQ
        # preprocess images
        batched_input, boxes_list, object_name_list = prepare_sam_data(images=images, objects_list=objects,
                                                                       Hs=Hs, Ws=Ws,
                                                                       resize_size=sam.image_encoder.img_size)
        # forward sam
        batched_output = sam(batched_input, multimask_output=False)
        masks_list = [output['masks'].cpu().numpy() for output in batched_output]

        # >>> Inpainting: Stable Diffusion
        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16).to("cuda")
        inpaint_images = [img.resize((512, 512)) for img in images]
        inpaint_masks = []
        for masks in masks_list:
            mask = masks[0][0]
            mask = Image.fromarray(mask).resize((512, 512))
            inpaint_masks.append(mask)
        after_inpaint_images = inpaint_pipe(image=inpaint_images, prompt=[''] * args.batch_size, mask_image=inpaint_masks).images
        for ipt_img, ipt_mask, aft_ipt_img, h, w in zip(inpaint_images, inpaint_masks, after_inpaint_images, Hs, Ws):
            run.log({'inpaint': [wandb.Image(ipt_img.resize((w, h)), caption='raw image'),
                                 wandb.Image(ipt_mask.resize((w, h)), caption='inpaint mask'),
                                 wandb.Image(aft_ipt_img.resize((w, h)), caption='inpainted image')]})

        # >>> Visualize: Wandb
        wandb_visualize(images, boxes_list, masks_list, object_name_list)

        from IPython import embed
        embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--sam_checkpoint", type=str, help="path to checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, help="path to checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")
    parser.add_argument("--split", default=",", type=str, help="split for text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    device = "cuda"

    run = wandb.init('Tag2Text & Grounded DINO & HQ-SAM')
    main()
    wandb.finish()
