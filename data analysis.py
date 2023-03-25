import json
import os

train_data = json.load(open(os.path.join('./data/', 'MR_Train.json'), 'r', encoding='utf-8'))
test_data = json.load(open(os.path.join('./data/', 'MR_Test.json'), 'r', encoding='utf-8'))
sst5_label_dict = {'very negative': 'terrible', 'negative': 'bad', 'neutral': 'okay', 'positive': 'good',
                   'very positive': 'perfect'}
sst2_label_dict = {'positive': 'great', 'negative': 'terrible'}

result = []
for raw_data in test_data:
    label = sst2_label_dict[raw_data['label']]
    text = raw_data['text']
    item = {
        "text": text,
        "label": label
    }
    result.append(item)

with open('data/MR_Test.json', 'w') as w:
    json.dump(result, w)
