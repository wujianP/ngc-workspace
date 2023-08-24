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
        tmp = 'this is a test, this is a test, this is a test, this is a test, this is a test, this is a test, this is a test, '
        inputs = self.tokenizer([ipt, opt, tmp], padding=True, max_length=self.tokenizer.model_max_length)
        data = {
            "input_ids": inputs['input_ids'][0],
            "labels": inputs['input_ids'][1],
            "attention_mask": inputs['attention_mask'][1],
        }
        return data
