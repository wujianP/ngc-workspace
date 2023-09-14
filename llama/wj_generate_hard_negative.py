import argparse
import time
import torch
import os

import numpy as np

from dataset import CocoShardedDataset
from llama import Llama
from torch.utils.data import DataLoader


def process_captions(captions, prompt):
    dialogs = []
    for caption in captions:
        dialog = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": 'A brown horse is grazing grass near a red house.'},
            {"role": "assistant", "content": """Reason: (swap two adjectives): Result: A red horse is grazing grass near a brown house."""},
            {"role": "user", "content": 'An angled view of a beautifully decorated bathroom.'},
            {"role": "assistant", "content": """Reason: (alter an adjective): Result: An angled view of a poorly decorated bathroom."""},
            {"role": "user", "content": 'A bus is parked by a bench at night.'},
            {"role": "assistant", "content": """Reason: (alter a noun): Result: A bus is parked by a bench at night."""},
            {"role": "user", "content": 'Two bicycles and a woman walking in front of a shop'},
            {"role": "assistant", "content": """Reason: (swap two nouns): Result: Two woman and a bicycles walking in front of a shop"""},
            {"role": "user", "content": 'A sink and a toilet inside a small bathroom.'},
            {"role": "assistant", "content": """Reason: (alter a preposition): Result: A sink and a toilet outside a small bathroom."""},
            {"role": "user", "content": 'The two people are sitting down the beach.'},
            {"role": "assistant", "content": """Reason: (alter a verb): Result: The two people are sitting down the beach."""},
            {"role": "user", "content": 'Two cats sleep on the bed while a man stands'},
            {"role": "assistant", "content": """Reason: (swap two verbs): Result: Two cats stand on the bed while a man sleeps"""},
            {"role": "user", "content": caption}
        ]
        dialogs.append(dialog)
    return dialogs


def my_collate_fn(batch):
    captions, ann_ids = [], []
    for item in batch:
        captions.append(item[0])
        ann_ids.append(item[1])
    return captions, ann_ids


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    dataset = CocoShardedDataset(image_root=args.images_path,
                          json=args.annotations_path,
                          gpu_id=args.gpu)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            collate_fn=my_collate_fn,
                            num_workers=args.num_workers)

    result_to_save = {'configure': vars(args)}
    total_iters = len(dataloader)
    for cur_idx, (captions, ann_ids) in enumerate(dataloader):
        start_time = time.time()

        from IPython import embed
        embed()

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
            filename = os.path.join(args.output_dir, args.filename + f'_gpu{args.gpu}.pth')
            torch.save(result_to_save, filename)

        for cap, result in zip(captions, results):
            print(f'Input: {cap}')
            output = result['generation']['content']
            print(f'Output: {output}')
            print()

        # output print
        i = np.random.randint(0, args.batch_size)
        print(f'Input: {captions[i]}')
        for i in range(len(caption_list)):
            print(f'{i}. {caption_list[i]}')
        print(f'ITERS: [{cur_idx+1} / {total_iters}] BATCH TIME: {(time.time() - start_time):.3f}')
        print("\n==================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('LaMma2')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--images_path', type=str, default='/DDN_ROOT/wjpeng/dataset/coco2014/images/train2014')
    parser.add_argument('--annotations_path', type=str, default='/DDN_ROOT/wjpeng/dataset/coco2014/annotations/captions_train2014.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--filename', type=str, required=True)

    parser.add_argument('--ckpt_dir', type=str, default='/discobox/wjpeng/weights/llama2/7B/')
    parser.add_argument('--tokenizer_path', type=str, default='/discobox/wjpeng/weights/llama2/tokenizer.model')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--max_batch_size', type=int, default=32)
    parser.add_argument('--max_gen_len', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.9)

    args = parser.parse_args()
    main(args)

