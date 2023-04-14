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

l = [92.6, 56.1, 98.0, 75.5, 92.7, 88.2, 97.2, 92.0]
print(mean(l))
