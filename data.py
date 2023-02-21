import json
import os
from functools import partial

import numpy as np
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
            # 1 No label
            # dataset.append((label_list + sep_token + tokens, label_id))
            dataset.append((tokens ,label_id))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


# Make tokens for every batch
def my_collate(batch, tokenizer, num_classes, method_name):
    tokens, label_ids = map(list, zip(*batch))

    text_ids = tokenizer(tokens,
                         padding=True,
                         max_length=512,
                         truncation=True,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')

    # the text_ids includes input_ids,token_type_ids,attention_mask
    positions = torch.zeros_like(text_ids['input_ids'])
    # 2 No label
    # positions[:, num_classes:] = torch.arange(0, text_ids['input_ids'].size(1) - num_classes)
    # positions[:]=torch.arange(0, text_ids['input_ids'].size(1))
    # text_ids['position_ids'] = positions

    # print(2,text_ids['attention_mask'])
    # print(3,text_ids['input_ids'])

    if (method_name == 'cls_explain'):
        start_indexs = []
        end_indexs = []
        lengths = []
        span_masks = []
        for i in text_ids['input_ids']:
            lengths.append(torch.count_nonzero(i).item())
        max_sentence_length = max(lengths)

        # 2+num_classes cas = [cls,lable1,lable2,sep,tokens....,sep]
        for i in range(2 + num_classes, max_sentence_length - 2):
            for j in range(i, max_sentence_length - 2):
                start_indexs.append(i)
                end_indexs.append(j)

        # 102 means [SEP]
        for index in range(len(batch)):
            span_mask = []
            # print(text_ids['input_ids'][index])
            sep_token_first = text_ids['input_ids'][index].tolist().index(102)
            sep_token = text_ids['input_ids'][index].tolist().index(102, sep_token_first + 1)
            for start_index, end_index in zip(start_indexs, end_indexs):
                if 2 + num_classes <= start_index <= lengths[index] - 2 and 2 + num_classes <= end_index <= lengths[
                    index] - 2 and (
                        start_index > sep_token or end_index < sep_token):
                    span_mask.append(0)
                else:
                    span_mask.append(1e6)
            span_masks.append(span_mask)

        text_ids['lengths'] = torch.tensor(np.array(lengths))
        text_ids['start_indexs'] = torch.tensor(np.array(start_indexs))
        text_ids['end_indexs'] = torch.tensor(np.array(end_indexs))
        text_ids['span_masks'] = torch.tensor(np.array(span_masks))
        # print('kk', text_ids['lengths'], text_ids['start_indexs'], text_ids['end_indexs'], text_ids['span_masks'])
    return text_ids, torch.tensor(label_ids)


# Load dataset
def load_data(dataset, data_dir, tokenizer, train_batch_size, test_batch_size, model_name, method_name, workers):
    if dataset == 'sst2':
        train_data = json.load(open(os.path.join(data_dir, 'SST2_全球公认_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST2_全球公认_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'sst5':
        train_data = json.load(open(os.path.join(data_dir, 'SST5_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST5_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4}
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