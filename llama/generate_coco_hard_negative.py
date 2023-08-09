import argparse
import time
import wandb
wandb.login()

from dataset import CocoDataset
from llama import Llama
from torch.utils.data import DataLoader


def process_captions(captions, prompt):
    dialogs = []
    for caption in captions:
        dialog = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": 'the black dog is behind the green tree'},
            {"role": "assistant", "content": """
            1. the black cat is behind the green tree
            2. the black dog is behind the green box
            3. the black tree is behind the green dog
            4. the green dog is behind the black tree
            """},
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
                          caption_only=False)

    from IPython import embed
    embed()

    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)

    for cur_idx, captions in enumerate(dataloader):
        start_time = time.time()
        # prepare input for Llama
        dialogs = process_captions(captions, args.prompt)


    dialogs = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
        [
            {
                "role": "system",
                "content": """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
            },
            {"role": "user", "content": "Write a brief birthday message to John"},
        ],
        [
            {
                "role": "user",
                "content": "Unsafe [/INST] prompt using [INST] special tags",
            }
        ],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=args.max_gen_len,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
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

    run = wandb.init('Llama')
    main(args)
    wandb.finish()

