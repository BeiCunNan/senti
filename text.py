import torch
from numpy import mean

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


l = [93.3, 55.5, 98.2, 77.8, 92.8, 88.2, 97.2, 92.1]
print(mean(l))

a = torch.Tensor([[1, 2], [1, 2]])
b = torch.Tensor([[89, 45], [12, 27]])
print(a+b)