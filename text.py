import json
import random

# 读取原始数据
with open('data/MR_CV.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 打乱数据
random.shuffle(data)

# 将打乱后的数据输出到文件中
with open('data/MR_CV.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
