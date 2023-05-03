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

l = [0.972, 0.972, 0.972, 0.972, 0.97, 0.974, 0.976, 0.976, 0.968, 0.972, 0.972, 0.968, 0.972, 0.97, 0.97, 0.97, 0.972, 0.968, 0.97, 0.976, 0.976, 0.974, 0.974, 0.97, 0.974, 0.972]
print(mean(l))
