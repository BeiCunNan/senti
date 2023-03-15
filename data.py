import json
import os
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader


# Make MyDataset
class MyDataset(Dataset):
    def __init__(self, raw_data, label_dict, tokenizer, model_name):
        label_list = list(label_dict.keys())
        split_token = '[SEP]'
        QUERY = 'please choose a correct sentiment category from { ' + ', '.join(label_list) + ' }'
        # print(len(QUERY.split(' ')))
        dataset = list()
        for data in raw_data:
            tokens = (QUERY + split_token + data['text'].lower()).split(' ')
            cls_sens = QUERY
            query_sens = data['text'].lower().split(' ')
            label_ids = label_dict[data['label']]
            dataset.append((tokens, label_ids, cls_sens, query_sens))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


# Make tokens for every batch
def my_collate(batch, tokenizer, num_classes, method_name):
    tokens, label_ids, cls_sens, query_sens = map(list, zip(*batch))

    text_ids = tokenizer(tokens,
                         padding=True,
                         max_length=512,
                         truncation=True,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    cls_ids = tokenizer(tokens,
                        padding=True,
                        max_length=512,
                        truncation=True,
                        is_split_into_words=True,
                        add_special_tokens=True,
                        return_tensors='pt')
    query_ids = tokenizer(tokens,
                          padding=True,
                          max_length=512,
                          truncation=True,
                          is_split_into_words=True,
                          add_special_tokens=True,
                          return_tensors='pt')
    return text_ids, torch.tensor(label_ids), cls_ids, query_ids


# Load dataset
def load_data(dataset, data_dir, tokenizer, train_batch_size, test_batch_size, model_name, method_name, workers):
    if dataset == 'sst2':
        train_data = json.load(open(os.path.join(data_dir, 'SST2_全球公认_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST2_全球公认_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'sst5':
        train_data = json.load(open(os.path.join(data_dir, 'SST5_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST5_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'very negative': 0, 'negative': 1, 'neutral': 2, 'positive': 3, 'very positive': 4}
    elif dataset == 'cr':
        train_data = json.load(open(os.path.join(data_dir, 'CR_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'CR_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'subj':
        train_data = json.load(open(os.path.join(data_dir, 'SUBJ_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SUBJ_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'subjective': 0, 'objective': 1}
    elif dataset == 'pc':
        train_data = json.load(open(os.path.join(data_dir, 'PC_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'PC_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'mr':
        train_data = json.load(open(os.path.join(data_dir, 'MR_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'MR_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'trec':
        train_data = json.load(open(os.path.join(data_dir, 'TREC_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'TREC_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'description': 0, 'entity': 1, 'abbreviation': 2, 'human': 3, 'location': 4, 'numeric': 5}
    else:
        raise ValueError('unknown dataset')

    trainset = MyDataset(train_data, label_dict, tokenizer, model_name)
    testset = MyDataset(test_data, label_dict, tokenizer, model_name)

    collate_fn = partial(my_collate, tokenizer=tokenizer, num_classes=len(label_dict), method_name=method_name)
    train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn,
                                  pin_memory=True)
    test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn,
                                 pin_memory=True)
    return train_dataloader, test_dataloader
