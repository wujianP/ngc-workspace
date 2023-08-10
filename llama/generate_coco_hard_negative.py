import argparse
import time
import wandb
import torch
import os

import numpy as np

from dataset import CocoDataset
from llama import Llama
from torch.utils.data import DataLoader


def process_captions(captions, prompt):
    dialogs = []
    for caption in captions:
        dialog = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": 'A black car is near someone riding a bike.'},
            {"role": "assistant", "content": """
            1. A black dog is near someone riding a bike.
            2. A black car is far away from someone riding a bike.
            3. A white car is near someone riding a bike.
            4. Many black cars are near someone riding a bike.
            5. A black bike is near someone driving a car."""},
            {"role": "user", "content": caption}
        ]
        dialogs.append(dialog)
    return dialogs


def main(args):
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    dataset = CocoDataset(image_root=args.images_path,
                          json=args.annotations_path)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)

    result_to_save = {'configure': vars(args)}
    total_iters = len(dataloader)
    for cur_idx, (ann_ids, captions) in enumerate(dataloader):
        start_time = time.time()

        # prepare input for Llama
        dialogs = process_captions(captions, args.prompt)

        # llama forward
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # post process result and save result
        for result, ann_id in zip(results, ann_ids):
            # preprocess
            output = result['generation']['content']
            output = output.strip().split('\n')[1:]
            caption_list = []
            for cap in output:
                caption_list.append(cap[3:].strip())
            # save
            result_to_save[ann_id] = caption_list
        if cur_idx % args.save_freq == 0:
            filename = os.path.join(args.output_dir, args.filename)
            torch.save(result_to_save, filename)

        # output print
        i = np.random.randint(0, args.batch_size)
        print(f'Input: {captions[i]}')
        print(f"Output: {results[i]['generation']['content']}")
        print(f'ITERS: [{cur_idx+1} / {total_iters}] BATCH TIME: {(time.time() - start_time):.3f}')
        print("\n==================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('LaMma2')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--images_path', type=str, default='/discobox/wjpeng/dataset/coco2014/images/train2014')
    parser.add_argument('--annotations_path', type=str, default='/discobox/wjpeng/dataset/coco2014/annotations/captions_train2014.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--filename', type=str, required=True)

    parser.add_argument('--ckpt_dir', type=str, default='/discobox/wjpeng/weights/llama2/llama-2-7b-chat/')
    parser.add_argument('--tokenizer_path', type=str, default='/discobox/wjpeng/weights/llama2/tokenizer.model')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--max_batch_size', type=int, default=128)
    parser.add_argument('--max_gen_len', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.9)

    args = parser.parse_args()
    wandb.login(key='8cff0498531e0409db5f3c43b52a26b0d068f2dc')
    run = wandb.init('Llama')
    main(args)
    wandb.finish()

