import torch
from transformers import SegformerFeatureExtractor, AutoImageProcessor, SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from dataset import RawFrameDataset
from torch import nn


@torch.no_grad()
def main(args):
    feature_extractor = SegformerFeatureExtractor.from_pretrained(args.model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_path).cuda()

    dataset = RawFrameDataset(path_file=args.data_path_file,
                              feature_extractor=feature_extractor)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False)

    for batch_idx, (images, heights, widths, paths) in enumerate(dataloader):
        # forward pass
        images = images.cuda()
        outputs = model(images)
        logits = outputs.logits

        from IPython import embed
        embed()

        # rescale logits to original image size
        for i in range(args.batch_size):
            height, weight = heights[i], widths[i]
            logits[i] = nn.functional.interpolate(logits[i], size=(height, weight), mode='bilinear', align_corners=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='nvidia/segformer-b0-finetuned-ade-512-512')
    parser.add_argument('--data_path_file', type=str, default='/discobox/wjpeng/dataset/k400/ann/rawframe_list.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    main(args)
