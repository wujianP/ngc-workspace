import argparse
import os.path

import torch
import json

from dataset import InpaintedDataset
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

import wandb

wandb.login()
run = wandb.init('BLIP Image Captioning')


def my_collate_fn(batch):
    images, inpainted_images, metadatas = [], [], []
    for item in batch:
        images.append(item[0])
        inpainted_images.append(item[1])
        metadatas.append(item[2])
    return [images, inpainted_images, metadatas]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--data_ann', type=str)
    parser.add_argument('--outputs', type=str)
    parser.add_argument('--model_checkpoint', type=str)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    dataset = InpaintedDataset(data_root=args.data_root, ann=args.data_ann)

    # blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=args.model_checkpoint)
    # blip = Blip2ForConditionalGeneration.from_pretrained(
    #     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32,
    #     cache_dir=args.model_checkpoint
    #
    # )
    # blip = blip.cuda()

    instruct_blip = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b",
                                                                         torch_dtype=torch.float16,
                                                                         cache_dir='/discobox/wjpeng/weights/instruct-blip')
    instruct_blip = instruct_blip.cuda()
    instruct_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b",
                                                               cache_dir='/discobox/wjpeng/weights/instruct-blip')

    for idx, (image, inpainted_image, metadata, path) in enumerate(dataset):

        # Instruct BLIP
        tags = metadata['original_tags']
        inpainted_tags = metadata['removed_tags'][0]

        inputs = instruct_processor(images=image,
                                    text='Write a detailed description of this image.',
                                    return_tensors="pt")
        inputs = inputs.to("cuda", torch.float16)

        generated_ids = instruct_blip.generate(**inputs,
                                               do_sample=False,
                                               num_beams=5,
                                               max_new_tokens=256,
                                               min_length=1,
                                               top_p=0.9,
                                               repetition_penalty=1.5,
                                               length_penalty=1.0,
                                               temperature=1, )

        caption = instruct_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(caption)
        torch.cuda.empty_cache()

        #  Visualize
        run.log({'Caption': [wandb.Image(image, caption=f'InstructBLIP: {caption}'),
                             wandb.Image(inpainted_image)]})

        # Save
        with open(os.path.join(path, 'captions.json'), 'w', encoding='utf-8') as file:
            ret = {
                'instruct-blip': caption
            }
            json.dump(ret, file, ensure_ascii=False, indent=4)

        print(f'{idx+1}/120')
