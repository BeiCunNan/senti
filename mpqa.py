import json

mpqa_label_dict = {'0': 'positive', '1': 'negative', '2': 'other'}

result = []

with open(r'C:\Users\大白菜\Desktop\ie-datasets\ie_label.txt') as label_file, open(
        r'C:\Users\大白菜\Desktop\ie-datasets\ie_word.txt') as word_file:
    for label, sentence in zip(label_file, word_file):
        label = mpqa_label_dict[label.strip()]
        sentence = sentence.strip()
        item = {
            "text": sentence,
            "label": label
        }
        result.append(item)
result_test = result[-300:]
result_train = result[:-300]
#
#
# with open('E:\data\Idiom_Sentiment_Analysis-master\\sentences.csv', 'r') as f:
#     reader = csv.reader(f)
#     rows = list(reader)
#     test_rows = random.sample(rows, 500)
#     train_rows = [row for row in rows if row not in test_rows]
#
#     for row in test_rows:
#         label = mpqa_label_dict[row[2]]
#         text = row[1]
#         item = {
#             "text": text,
#             "label": label
#         }
#         result_test.append(item)
#
#     for row in train_rows:
#         label = mpqa_label_dict[row[2]]
#         text = row[1]
#         item = {
#             "text": text,
#             "label": label
#         }
#         result_train.append(item)
# #
# #
with open('data/IE_Test.json', 'w') as w:
    json.dump(result_test, w)
#
with open('data/IE_Train.json', 'w') as w:
    json.dump(result_train, w)
