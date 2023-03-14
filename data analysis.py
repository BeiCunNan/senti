import json
import os

train_data = json.load(open(os.path.join('./data/', 'SST5_Train.json'), 'r', encoding='utf-8'))
test_data = json.load(open(os.path.join('./data/', 'SST5_Test.json'), 'r', encoding='utf-8'))
train_label_dict = {'zero': 'very negative', 'one': 'negative', 'two': 'neutral', 'three': 'positive',
                    'four': 'very positive'}
test_label_dict = {'zero': 'very negative', 'one': 'negative', 'two': 'neutral', 'three': 'positive',
                   'four': 'very positive'}

result = []
for raw_data in test_data:
    label = train_label_dict[raw_data['label']]
    text = raw_data['text']
    item = {
        "text": text,
        "label": label
    }
    result.append(item)

with open('./data/SST5_Test.json', 'w') as w:
     json.dump(result, w)
