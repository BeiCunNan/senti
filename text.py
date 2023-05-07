# # 读取原始数据
# with open('data/MR_CV.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
#
# # 打乱数据
# random.shuffle(data)
#
# # 将打乱后的数据输出到文件中
# with open('data/MR_CV.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

from numpy import mean

l = [87.8	,92.3,	97.8	,97.1,	91.8,	92.4,	54.8	,75.0]
print(mean(l))

# import csv
# import json
#
# sst2_label_dict = {'0': 'negative', '1': 'positive'}
#
# result = []
#
# # 打开 TSV 文件
# with open("train.tsv", "r", encoding='utf-8') as f:
#     # 创建并返回一个 CSV 读取器对象
#     reader = csv.reader(f)
#     # 将读取的数据存储在列表中
#     for row in f:
#         m = row.split('	')
#         item = {
#             "text": m[1].replace("\n", ""),
#             "label": sst2_label_dict[m[0]]
#         }
#         result.append(item)
#
# with open('data/SST2_Train.json', 'w') as w:
#     json.dump(result, w)
