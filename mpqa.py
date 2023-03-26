import json
import random

mpqa_label_dict = {'positive': 'positive', 'negative': 'negative', 'other': 'other'}

result_train = []
result_test = []

import csv

with open('E:\data\Idiom_Sentiment_Analysis-master\\sentences.csv', 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)
    test_rows = random.sample(rows, 500)
    train_rows = [row for row in rows if row not in test_rows]

    for row in test_rows:
        label = mpqa_label_dict[row[2]]
        text = row[1]
        item = {
            "text": text,
            "label": label
        }
        result_test.append(item)

    for row in train_rows:
        label = mpqa_label_dict[row[2]]
        text = row[1]
        item = {
            "text": text,
            "label": label
        }
        result_train.append(item)
#
#
with open('data/IE_Test.json', 'w') as w:
    json.dump(result_test, w)

with open('data/IE_Train.json', 'w') as w:
    json.dump(result_train, w)
