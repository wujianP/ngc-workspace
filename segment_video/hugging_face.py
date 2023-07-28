from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from .dataset import RawFrameDataset


def main(args):
    image_processor = AutoImageProcessor.from_pretrained(args.model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_path)

    dataset = RawFrameDataset(path_file=args.data_path_file)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    from IPython import embed
    embed()

    inputs = image_processor(images=dataset[0], return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
    list(logits.shape)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='nvidia/segformer-b0-finetuned-ade-512-512')
    parser.add_argument('--data_path_file', type=str, default='/discobox/wjpeng/dataset/k400/ann/rawframe_list.txt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    main(args)
