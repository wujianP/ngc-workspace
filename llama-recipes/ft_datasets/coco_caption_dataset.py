import torch


def get_coco_caption_dataset(dataset_config, tokenizer, split):
    return CoCoCaptionDataset(dataset_config, tokenizer, split)


class CoCoCaptionDataset:
    def __init__(self, dataset_config, tokenizer, split):
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.data = torch.load(split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ipt = self.data[idx]['input']
        opt = self.data[idx]['output']
        input_ids = self.tokenizer(ipt)['input_ids']
        labels = self.tokenizer(opt)['input_ids']
        attention_mask = self.tokenizer(opt)['attention_mask']
        data = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        return data
