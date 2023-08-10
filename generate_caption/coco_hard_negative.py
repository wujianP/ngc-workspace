"""
This is a script that merge several frame-level captions into a video-level captions
"""

import argparse
import torch
import time
import csv

from fastchat.model import load_model
from dataset import CocoDataset
from torch.utils.data import DataLoader


def prepare_prompts(prefix, captions):
    prompts = []
    for cap in captions:
        prompt = prefix + cap
        prompts.append(prompt)
    return prompts

@torch.inference_mode()
def main(args):
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
    coco_dataset = CocoDataset(image_root=args.images_path,
                               json=args.annotations_path,
                               transforms=None,
                               caption_only=True)

    coco_dataloader = DataLoader(dataset=coco_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=8,
                                 pin_memory=True,
                                 shuffle=True)

    # do inference
    total_iters = len(coco_dataloader)
    for cur_iter, (captions) in enumerate(coco_dataloader):
        start_time = time.time()
        # prepare prompts

        from IPython import embed
        embed()

        prompts = prepare_prompts(captions)
        # tokenize
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['attention_mask'] = inputs['attention_mask'].cuda()

        # forward
        output_sequences = model.generate(
            input_ids=inputs['input_ids'],
            do_sample=True,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        # decode
        outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True,
                                         spaces_between_special_tokens=False)

        # post-process
        if model.config.is_encoder_decoder:
            pass
        else:
            outputs = [output.split('\nOutput: ')[-1] for output in outputs]

        end_time = time.time()
        batch_time = end_time - start_time

        # save file
        with open(args.save_path, "a", newline="") as f:
            writer = csv.writer(f)
            for i in range(args.batch_size):
                output = outputs[i].rstrip("\n")

        print('Input:' + captions[0])
        print('Output:' + outputs[0])
        print(f"Iteration:[ {cur_iter + 1}/{total_iters} ], Batch time:{batch_time:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data and Model
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.3")
    parser.add_argument('--images_path', type=str, default='/discobox/wjpeng/dataset/coco2014/images/train2014')
    parser.add_argument('--annotations_path', type=str, default='/discobox/wjpeng/dataset/coco2014/annotations/captions_train2014.json')
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

    main(args)
