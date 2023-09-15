"""
This is a script that merge several frame-level captions into a video-level captions
"""

import argparse
import os.path
import random

import torch
import time

from fastchat.model import load_model
from dataset import NegCLIPCocoDataset
from torch.utils.data import DataLoader, Subset


def prepare_prompts(prefix, captions):
    prompts = []
    for cap in captions:
        cap = cap.strip()
        prompt = f'{prefix}\nInput: {cap}\nOutput: '
        prompts.append(prompt)
    return prompts


def my_collate_fn(batch):
    caption_list, image_path_list, hard_image_path_list = [], [], []
    for item in batch:
        caption_list.append(item[0])
        image_path_list.append(item[1])
        hard_image_path_list.append(item[2])
    return caption_list, image_path_list, hard_image_path_list


@torch.inference_mode()
@torch.no_grad()
def main():
    # make dir
    os.makedirs(args.outputs, exist_ok=True)
    output_file = os.path.join(args.outputs, f'job{args.job_id:02d}_{args.job_num}.pth')
    print(f'File outputs to: {output_file}')

    # load model
    model, tokenizer = load_model(
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        revision=args.revision, )

    # load data
    coco_dataset = NegCLIPCocoDataset(image_root=args.images_path,
                                      ann=args.annotations_path)

    total_len = len(coco_dataset)
    split_len = total_len // args.job_num
    start_idx = args.job_id * split_len
    end_idx = start_idx + split_len
    if args.job_id == (args.job_num - 1):
        end_idx = total_len
    split_dataset = Subset(coco_dataset, range(start_idx, end_idx))

    print(f'job id: {args.job_id} job num: {args.job_num}  start idx: {start_idx} end idx: {end_idx}')

    coco_dataloader = DataLoader(dataset=split_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=8,
                                 pin_memory=True,
                                 shuffle=False,
                                 collate_fn=my_collate_fn)

    # do inference
    ret_list = []
    total_iters = len(coco_dataloader)
    for cur_iter, (caption_list, image_path_list, hard_image_path_list) in enumerate(coco_dataloader):
        # from IPython import embed
        # embed()
        start_time = time.time()
        # >>> Name Entities Recognition >>>
        sen2ne_template = """Task: Identify all the nouns and noun phrase in the given sentences. Examples:
        Input: A white door in a kitchen with teal walls.
        Output: door,kitchen,wall
        Input: Cat sitting on the hood of a car on a winter day.
        Output: cat,car,hood,day
        Input: Two dogs are relaxing beside a man on a sidewalk.
        Output: dog,man,sidewalk
        Input: A group of young children wearing costumes standing in line.
        Output: children,costumes,line
        Input: Two pastries on a plate with chocolate milk.
        Output: pastries,plate,chocolate milk
        Input: A toilet stall with exposed brick and free standing tank.
        Output: toilet stall,brick,tank
        Input: A dog with a wine glass being held to its face
        Output: dog,wine glass,face
        Input: Two men play a game on a Wii console in a living room.
        Output: men,game,Wii console,living room
        Input: Two young children in colorful clothes are playing near a door.
        Output: children,clothes,door
        Input: A bus turning a corner at an intersection near a motorcycle.
        Output: bus,corner,intersection,motorcycle
        Input: A man is standing near the street with a surfboard.
        Output: man,street,surfboard
        Input: A grey bird perched on a tree next to a body of water.
        Output: bird,tree,body,water
        Input: Two men sitting in a sailboat with two sails.
        Output: men,sailboat,sails
        Input: A passenger train that is traveling down the tracks.
        Output: passenger train,tracks
        Input: A man on a motorcycle holding one arm in the air.
        Output: man,motorcycle,arm,air
        Input: Several elephants standing in mud while people watched them from a building on a cliff.
        Output: elephants,mud,people,building,cliff"""

        sen2ne_prompts = prepare_prompts(sen2ne_template, caption_list)

        sen2ne_inputs = tokenizer(sen2ne_prompts, padding=True, truncation=True, return_tensors="pt")
        sen2ne_inputs['input_ids'] = sen2ne_inputs['input_ids'].cuda()
        sen2ne_inputs['attention_mask'] = sen2ne_inputs['attention_mask'].cuda()

        sen2ne_output_sequences = model.generate(
            input_ids=sen2ne_inputs['input_ids'],
            do_sample=True,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        sen2ne_outputs = tokenizer.batch_decode(sen2ne_output_sequences,
                                                skip_special_tokens=True,
                                                spaces_between_special_tokens=False)
        for idx, ne in enumerate(sen2ne_outputs):
            ne = ne.strip().split(',')
            random.shuffle(ne)
            sen2ne_outputs[idx] = ','.join(ne)

        torch.cuda.empty_cache()

        # >>> Generate sentence with named entities
        ne2sen_templates = """In this task, I will give you some words, and you will imagine a scene that contains all the words, you can add more details.
        Input: door,kitchen,wall
        Output: A white door in a kitchen with teal walls.
        Input: cat,car,hood
        Output: Cat sitting on the hood of a car on a winter day.
        Input: dog,man,sidewalk
        Output: Two dogs are relaxing beside a man on a sidewalk.
        Input: children,costumes,line
        Output: A group of young children wearing costumes standing in line.
        Input: chocolate milk,pastries,plate
        Output: Two pastries on a plate with chocolate milk.
        Input: toilet stall,brick,tank
        Output: A toilet stall with exposed brick and free standing tank.
        Input: wine glass,dog,face
        Output: A dog with a wine glass being held to its face
        Input: living room,game,men,Wii console
        Output: Two men play a game on a Wii console in a living room.
        Input: clothes,door,children
        Output: Two young children in colorful clothes are playing near a door.
        Input: bus,intersection,corner,motorcycle
        Output: A bus turning a corner at an intersection near a motorcycle.
        Input: street,man,surfboard
        Output: A man is standing near the street with a surfboard.
        Input: bord,water,body,tree
        Output: A grey bird perched on a tree next to a body of water.
        Input: sailboat,sail,men
        Output: Two men sitting in a sailboat with two sails.
        Input: tracks,passenger train
        Output: A passenger train that is traveling down the tracks.
        Input: motorcycle,man,arm
        Output: A man on a motorcycle holding one arm in the air.
        Input: elephants,mud,cliff,people,building
        Output: Several elephants standing in mud while people watched them from a building on a cliff."""

        ne2sen_prompts = prepare_prompts(ne2sen_templates, sen2ne_outputs)

        ne2sen_inputs = tokenizer(ne2sen_prompts, padding=True, truncation=True, return_tensors="pt")
        ne2sen_inputs['input_ids'] = ne2sen_inputs['input_ids'].cuda()
        ne2sen_inputs['attention_mask'] = ne2sen_inputs['attention_mask'].cuda()
        # > once
        ne2sen_output_sequences_1 = model.generate(
            input_ids=ne2sen_inputs['input_ids'],
            do_sample=True,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        ne2sen_outputs_1 = tokenizer.batch_decode(ne2sen_output_sequences_1,
                                                  skip_special_tokens=True,
                                                  spaces_between_special_tokens=False)

        torch.cuda.empty_cache()

        # twice
        ne2sen_output_sequences_2 = model.generate(
            input_ids=ne2sen_inputs['input_ids'],
            do_sample=True,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        ne2sen_outputs_2 = tokenizer.batch_decode(ne2sen_output_sequences_2,
                                                  skip_special_tokens=True,
                                                  spaces_between_special_tokens=False)

        torch.cuda.empty_cache()

        # >>> Save Results >>>
        for neg_cap_1, neg_cap_2, cap, img_path, hard_image_path in zip(ne2sen_outputs_1, ne2sen_outputs_2, caption_list, image_path_list, hard_image_path_list):
            ret = {
                'image_path': img_path,
                'neg_image_paths': hard_image_path,
                'caption': cap,
                'neg_captions': [neg_cap_1.strip(), neg_cap_2.strip()]
            }
            ret_list.append(ret)
        if cur_iter % args.save_freq == 0 or (cur_iter + 1) == total_iters:
            torch.save(ret_list, output_file)

        end_time = time.time()
        batch_time = end_time - start_time

        print(f'Input: {caption_list[0]}\n'
              f'Name Entities: {sen2ne_outputs[0]}\n'
              f'Hard Negative 1: {ne2sen_outputs_1[0]}'
              f'Hard Negative 2: {ne2sen_outputs_2[0]}')

        print(f"[ Job: {args.job_id}/{args.job_num} Iteration:{cur_iter + 1}/{total_iters}"
              f" ({100 * cur_iter / total_iters:.2f}%)], Batch time:{batch_time:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Job parallel
    parser.add_argument('--job_num', type=int, default=1)
    parser.add_argument('--job_id', type=int, default=0, help='begin from 0')

    # Data and Model
    parser.add_argument('--outputs', type=str, default='outputs')
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.3")
    parser.add_argument('--images_path', type=str, default='/DDN_ROOT/wjpeng/dataset/coco2014/images/train2014')
    parser.add_argument('--annotations_path', type=str,
                        default='/DDN_ROOT/wjpeng/dataset/coco2014/annotations/captions_train2014.json')
    parser.add_argument('--save_freq', type=int, default=50)

    # Hyper-parameters
    parser.add_argument('--batch-size', type=int, default=32)
    # parser.add_argument('--prompt-template', type=str, required=True)
    # Devices
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "xpu"], default="cuda",
                        help="The device type")
    parser.add_argument("--gpus", type=str, default=None, help="A single GPU like 1 or multiple GPUs like 0,2")
    parser.add_argument("--num-gpus", type=int, default=1)
    # LLM
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    # Others
    parser.add_argument("--revision", type=str, default="main", help="Hugging Face Hub model revision identifier")
    parser.add_argument("--max-gpu-memory", type=str, help="The maximum memory per gpu. Use a string like '13Gib'")
    parser.add_argument("--load-8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--cpu-offloading", action="store_true",
                        help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU")
    parser.add_argument("--gptq-ckpt", type=str, default=None,
                        help="Load quantized model. The path to the local GPTQ checkpoint.")
    parser.add_argument("--gptq-wbits", type=int, default=16, choices=[2, 3, 4, 8, 16],
                        help="#bits to use for quantization")
    parser.add_argument("--gptq-groupsize", type=int, default=-1,
                        help="Groupsize to use for quantization; default uses full row.")
    parser.add_argument("--gptq-act-order", action="store_true",
                        help="Whether to apply the activation order GPTQ heuristic")

    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    main()
