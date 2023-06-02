import json
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data import MyDataset, my_collate

SENTENCE = 'come on girl take it easy'

split = SENTENCE.split(' ')
length=len(split)
print(split)
"if they can get a foothold in europe they will have the chance to be bigger and better"
"如果他们能在欧洲站稳脚跟，他们将有机会变得更大更好。"
"soon it was dark and the christmas evening was in full swing"
"很快天黑了，圣诞之夜如火如荼"
"he was a strong healthy lad and as pleased as punch to be working with dad"
"他是一个强壮健康的小伙子，很高兴能和爸爸一起工作"
"other critics who could not get worked up about the play also admired the acting"
"其他对这部剧不满意的评论家也对演技赞不绝口。"
"not since the surprise smash hit of the year cast a rosy glow over shores vehicle"
"自从当年出人意料的热门歌曲在海岸车辆上投下玫瑰色的光芒以来"

# Get the model
bert = torch.load('./save/bert.pkl')
model = torch.load('./save/model.pkl')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model.eval()

# Data process
case_data = json.load(open(os.path.join('./data', 'IE_Case.json'), 'r', encoding='utf-8'))
label_dict = {'positive': 0, 'negative': 1, 'other': 2}
caseSet = MyDataset(case_data, label_dict, 'idiom')
collate_fn = partial(my_collate, tokenizer=tokenizer)
case_dataloader = DataLoader(caseSet, 1, shuffle=False, num_workers=0, collate_fn=collate_fn,
                             pin_memory=True)

# Get the result
for mrc_inputs, targets, text_inputs, mask_inputs, mask_index in tqdm(case_dataloader, disable=False,
                                                                      ascii=' >='):
    mrc_inputs = {k: v.to('cuda') for k, v in mrc_inputs.items()}
    text_inputs = {k: v.to('cuda') for k, v in text_inputs.items()}
    mask_inputs = {k: v.to('cuda') for k, v in mask_inputs.items()}
    targets = targets.to('cuda')

    predicts, bTSA, bFSA, text_tokens,cls_token = model(mrc_inputs, text_inputs, mask_inputs, mask_index)
    print(predicts)



# l2
cls=cls_token.repeat(length,1)
l2=torch.nn.functional.pairwise_distance(bFSA[0, 1:length+1, :], cls, p=2, eps=1e-06).cpu().detach().numpy()
l2=numpy.reshape(l2,(1,length))

# Draw the heatMap
X = torch.abs(bFSA[0, 1:length+1, :]).cpu().detach().numpy().T
plt.imshow(X, interpolation='bicubic', aspect='auto')

sentence = SENTENCE.split(' ')
plt.xticks(np.arange(len(sentence)), labels=sentence, rotation=45, rotation_mode="anchor", ha="right")
plt.yticks(np.arange(1))

plt.colorbar()
plt.show()
