import json
import os
from functools import partial

import torch
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data import MyDataset, my_collate


def get_p_value():
    MPTFWA = [88.5, 92.9, 97.4, 92.2, 98.0, 94.1, 56.1, 78.0]
    woFVSA = [87.8, 92.2, 97.0, 91.8, 97.6, 93.8, 55.0, 75.0]
    woTVSA = [88.2, 92.4, 97.1, 91.9, 97.6, 93.9, 55.2, 75.3]
    woTFWA = [87.9, 92.2, 96.9, 91.2, 97.6, 93.8, 54.2, 74.3]

    P1 = stats.ttest_ind(MPTFWA, woFVSA)
    P2 = stats.ttest_ind(MPTFWA, woTVSA)
    P3 = stats.ttest_ind(MPTFWA, woTFWA)
    print(P1, P2, P3)

    P1 = stats.ttest_rel(MPTFWA, woFVSA)
    P2 = stats.ttest_rel(MPTFWA, woTVSA)
    P3 = stats.ttest_rel(MPTFWA, woTFWA)

    print(P1, P2, P3)


def calculate_attention_weight():
    SENTENCE = 'He is scared stiff of it'
    MRC_SENTENCE = SENTENCE + ' [SEP] What class in { positive , negative , other } does this sentence have ?'
    PL_SENTENCE = SENTENCE + ' [SEP] This idiom is [MASK] .'

    context_split = SENTENCE.split(' ')
    mrc_split = MRC_SENTENCE.split(' ')
    pl_split = PL_SENTENCE.split(' ')

    context_length = len(context_split)
    mrc_length = len(mrc_split)
    pl_length = len(pl_split)

    print("Getting the model")
    model = torch.load('./save/model.pkl')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model.eval()
    print("Data processing")
    case_data = json.load(open(os.path.join('./data', 'IE_Case.json'), 'r', encoding='utf-8'))
    label_dict = {'positive': 0, 'negative': 1, 'other': 2}
    caseSet = MyDataset(case_data, label_dict, 'idiom')
    collate_fn = partial(my_collate, tokenizer=tokenizer)
    case_dataloader = DataLoader(caseSet, 1, shuffle=False, num_workers=0, collate_fn=collate_fn,
                                 pin_memory=True)
    print("Getting the result")
    for mrc_inputs, targets, text_inputs, mask_inputs, mask_index in tqdm(case_dataloader, disable=False,
                                                                          ascii=' >='):
        mrc_inputs = {k: v.to('cuda') for k, v in mrc_inputs.items()}
        text_inputs = {k: v.to('cuda') for k, v in text_inputs.items()}
        mask_inputs = {k: v.to('cuda') for k, v in mask_inputs.items()}
        targets = targets.to('cuda')
        predicts, aTSA, aFSA, mrc_tokens, mrc_CLS, bTSA, bFSA, context_tokens, text_CLS, cTSA, cFSA, pl_tokens, MASK = model(
            mrc_inputs, text_inputs, mask_inputs, mask_index)
        print(predicts)


if __name__ == '__main__':
    # get_p_value()
    calculate_attention_weight()
