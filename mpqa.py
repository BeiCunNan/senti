import json

mpqa_label_dict = {'0': 'bad', '1': 'good'}

result = []

import csv

with open('E:\data\各种各样的文本数据集\mpqa\\train.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        label = mpqa_label_dict[row[0]]
        text = row[1]
        item = {
            "text": text,
            "label": label
        }
        result.append(item)
#
#
with open('data/MPQA_Train.json', 'w') as w:
    json.dump(result, w)
