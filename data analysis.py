import json
import os

train_data = json.load(open(os.path.join('./data/', 'SST5_Train.json'), 'r', encoding='utf-8'))
test_data = json.load(open(os.path.join('./data/', 'SST5_Test.json'), 'r', encoding='utf-8'))
train_label_dict = {'zero': 0, 'one': 0, 'two': 0, 'three': 0, 'four': 0}
test_label_dict = {'zero': 0, 'one': 0, 'two': 0, 'three': 0, 'four': 0}

for raw_data in train_data:
    train_label_dict[raw_data['label']] = train_label_dict[raw_data['label']] + 1
for raw_data in test_data:
    test_label_dict[raw_data['label']] = test_label_dict[raw_data['label']] + 1

print('train_label_dict', train_label_dict)
print('test_label_dict', test_label_dict)
