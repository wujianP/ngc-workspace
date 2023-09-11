import argparse

from dataset import InpaintedDataset
from torch.utils.data import DataLoader


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

    for idx, (image, inpainted_image, metadata) in enumerate(dataset):

        from IPython import embed
        embed()

