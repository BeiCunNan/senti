import json
import os
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader

# Make MyDataset
class MyDataset(Dataset):
    def __init__(self, raw_data, label_dict, tokenizer, model_name):
        label_list = list(label_dict.keys())
        sep_token = ['[SEP]']
        dataset = list()
        for data in raw_data:
            tokens = data['text'].lower().split(' ')
            label_id = label_dict[data['label']]
            dataset.append((label_list + sep_token + tokens, label_id))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


# Make tokens for every batch
def my_collate(batch, tokenizer, num_classes):
    tokens, label_ids = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    positions = torch.zeros_like(text_ids['input_ids'])
    positions[:, num_classes:] = torch.arange(0, text_ids['input_ids'].size(1) - num_classes)
    text_ids['position_ids'] = positions

    return text_ids, torch.tensor(label_ids)


# Load dataset
def load_data(dataset, data_dir, tokenizer, train_batch_size, test_batch_size, model_name, workers):
    if dataset == 'sst2':
        train_data = json.load(open(os.path.join(data_dir, 'SST2_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST2_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'sst5':
        train_data = json.load(open(os.path.join(data_dir, 'SST5_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST5_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'one': 0, 'two': 1, 'three': 2, 'four': 3, 'five': 4}
    else:
        raise ValueError('unknown dataset')

    trainset = MyDataset(train_data, label_dict, tokenizer, model_name)
    testset = MyDataset(test_data, label_dict, tokenizer, model_name)

    collate_fn = partial(my_collate, tokenizer=tokenizer,  num_classes=len(label_dict))
    train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn,
                                  pin_memory=True)
    test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn,
                                 pin_memory=True)
    return train_dataloader, test_dataloader
