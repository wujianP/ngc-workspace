import argparse
import os

import json
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor, build_sam_hq
from segment_anything.utils.transforms import ResizeLongestSide

# Tag2Text
import sys
sys.path.append('Tag2Text')
from Tag2Text.models import tag2text
from Tag2Text import inference_tag2text
import torchvision.transforms as TS

from dataset import CoCoDataset
from torch.utils.data import DataLoader
from PIL import Image

import wandb

wandb.login()
run = wandb.init('Tag2Text & Grounded DINO & HQ-SAM')


def my_collate_fn(batch):
    images, Ws, Hs, paths = [], [], [], []
    for item in batch:
        images.append(item[0])
        Ws.append(item[1])
        Hs.append(item[2])
        paths.append(item[3])
    return [images, Ws, Hs, paths]


def generate_caption(raw_image, device):
    # unconditional image captioning
    if device == "cuda":
        inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
    else:
        inputs = processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def generate_tags(caption, split=',', max_tokens=100, model="gpt-3.5-turbo"):
    lemma = nltk.wordnet.WordNetLemmatizer()
    if openai_key:
        prompt = [
            {
                'role': 'system',
                'content': 'Extract the unique nouns in the caption. Remove all the adjectives. ' + \
                           f'List the nouns in singular form. Split them by "{split} ". ' + \
                           f'Caption: {caption}.'
            }
        ]
        response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
        reply = response['choices'][0]['message']['content']
        # sometimes return with "noun: xxx, xxx, xxx"
        tags = reply.split(':')[-1].strip()
    else:
        nltk.download(['punkt', 'averaged_perceptron_tagger', 'wordnet'])
        tags_list = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(caption)) if pos[0] == 'N']
        tags_lemma = [lemma.lemmatize(w) for w in tags_list]
        tags = ', '.join(map(str, tags_lemma))
    return tags


def check_caption(caption, pred_phrases, max_tokens=100, model="gpt-3.5-turbo"):
    object_list = [obj.split('(')[0] for obj in pred_phrases]
    object_num = []
    for obj in set(object_list):
        object_num.append(f'{object_list.count(obj)} {obj}')
    object_num = ', '.join(object_num)
    print(f"Correct object number: {object_num}")

    if openai_key:
        prompt = [
            {
                'role': 'system',
                'content': 'Revise the number in the caption if it is wrong. ' + \
                           f'Caption: {caption}. ' + \
                           f'True object number: {object_num}. ' + \
                           'Only give the revised caption: '
            }
        ]
        response = openai.ChatCompletion.create(model=model, messages=prompt, temperature=0.6, max_tokens=max_tokens)
        reply = response['choices'][0]['message']['content']
        # sometimes return with "Caption: xxx, xxx, xxx"
        caption = reply.split(':')[-1].strip()
    return caption


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
    dataset = CoCoDataset(image_root=args.data_root, json=args.data_ann)
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
        sam = build_sam_hq(checkpoint=args.sam_hq_checkpoint).cuda()
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
        tags = [tag.replace(' |', ',') for tag in tag2text_ret[0]]
        tag2text_captions = tag2text_ret[2]
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
        boxes_filt, scores, pred_phrases = get_grounding_output(
            model=grounding_dino_model,
            image=dino_images,
            caption=tags,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold
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
        # empty cache
        torch.cuda.empty_cache()

        # >>> Segmentation: inference sam >>>
        # preprocess images
        batched_input = prepare_sam_data(images=images, boxes=boxes_filt,
                                         Hs=Hs, Ws=Ws,
                                         resize_size=sam.image_encoder.img_size)

        from IPython import embed
        embed()
        # forward sam
        batched_output = sam(batched_input, multimask_output=False)
        masks_list = [output['masks'].cpu().numpy() for output in batched_output]

        # >>> Inpainting: inference stable diffusion or lama >>>

    # raw_image = image_pil.resize((384, 384))
    # raw_image = transform(raw_image).unsqueeze(0).to(device)
    raw_image, _, _, _ = dataset[0]
    raw_image_pt = transform(raw_image).unsqueeze(0).cuda()
    # Inference Tag2Text
    res = inference_tag2text.inference(image=raw_image_pt, model=tag2text_model, input_tag=args.user_specified_tags)

    # Currently ", " is better for detecting single tags
    # while ". " is a little worse in some case
    text_prompt, caption = res[0].replace(' |', ','), res[2]

    # run grounding dino model
    boxes_filt, scores, pred_phrases = get_grounding_output(
        model=grounding_dino_model,
        image=dino_images,
        caption=text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )

    # initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=args.sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    # use NMS to handle overlapped boxes
    print(f"Before NMS: {boxes_filt.shape[0]} boxes")
    nms_idx = torchvision.ops.nms(boxes_filt, scores, args.iou_threshold).numpy().tolist()
    boxes_filt = boxes_filt[nms_idx]
    pred_phrases = [pred_phrases[idx] for idx in nms_idx]
    print(f"After NMS: {boxes_filt.shape[0]} boxes")
    caption = check_caption(caption, pred_phrases)
    print(f"Revise caption with number: {caption}")

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.title('Tag2Text-Captioning: ' + caption + '\n' + 'Tag2Text-Tagging' + text_prompt + '\n')
    plt.axis('off')
    plt.savefig(
        os.path.join(args.output_dir, "automatic_label_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    save_mask_data(args.output_dir, caption, masks, boxes_filt, pred_phrases)
