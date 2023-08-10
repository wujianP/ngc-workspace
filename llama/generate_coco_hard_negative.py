import argparse
import time
import wandb

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
            1. A black bike is near someone riding a car.
            2. A white car is near someone riding a bike.
            3. A black car is far away from someone riding a bike.
            4. A black motorbike is near someone riding a bike."""},
            {"role": "user", "content": caption}
        ]
        dialogs.append(dialog)
    return dialogs


def main(args):
    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    dataset = CocoDataset(image_root=args.images_path,
                          json=args.annotations_path,
                          transforms=None,
                          caption_only=True)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)

    for cur_idx, captions in enumerate(dataloader):
        start_time = time.time()

        from IPython import embed
        embed()

        # prepare input for Llama
        prompt = """We will input a sentence, you need to change the one of the noun or noun phrase in the sentence to modify the objects it describes. If there are multiple nouns in the sentence, you can swap their positions. It's important to modify as few words as possible while keeping the overall sentence structure unchanged."""
        dialogs = process_captions(captions, prompt)
        dialogs = process_captions(captions, args.prompt)

        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        for caption, dialog, result in zip(captions, dialogs, results):
            # for msg in dialog:
            #     print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(f'Input: {caption}')
            print(
                f"Output: {result['generation']['content']}"
            )
            print("\n==================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('LaMma2')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--images_path', type=str, default='/discobox/wjpeng/dataset/coco2014/images/train2014')
    parser.add_argument('--annotations_path', type=str, default='/discobox/wjpeng/dataset/coco2014/annotations/captions_train2014.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--ckpt_dir', type=str, default='/discobox/wjpeng/weights/llama2/llama-2-7b-chat/')
    parser.add_argument('--tokenizer_path', type=str, default='/discobox/wjpeng/weights/llama2/tokenizer.model')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--max_batch_size', type=int, default=64)
    parser.add_argument('--max_gen_len', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.9)

    args = parser.parse_args()
    wandb.login(key='8cff0498531e0409db5f3c43b52a26b0d068f2dc')
    run = wandb.init('Llama')
    main(args)
    wandb.finish()

