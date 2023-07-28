from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class RawFrameDataset(Dataset):
    def __init__(self, path_file, transform=None):
        with open(path_file, 'r') as file:
            self.path_list = file.readlines()
        self.transform = transform

    def __getitem__(self, idx):
        path = self.path_list[idx].strip()
        with open(path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        f.close()

        if self.transform is not None:
            image = self.transform(image)

        return image, path

    def __len__(self):
        return len(self.path_list)
