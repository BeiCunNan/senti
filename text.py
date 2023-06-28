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

import numpy as np
import torch

a = torch.Tensor([1, 2, 3, 4])
b = torch.Tensor([2, 2, 3, 2])
c = torch.nn.functional.pairwise_distance(a, b, p=2, eps=1e-06)


def modified_sigmoid(x):
    def modified_sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def inverse_sigmoid(x):
        return 1 - modified_sigmoid(x)

    return inverse_sigmoid(x) / np.sum(inverse_sigmoid(x), axis=1, keepdims=True)

# print(modified_sigmoid([[1, 1], [2, 3]]))


def modified_softmax(x):
    def modified_softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / (np.sum(exp_values, axis=1, keepdims=True) + 1e-6)

    def inverse_softmax(x):
        return 1 - modified_softmax(x)

    return inverse_softmax(x) / np.sum(inverse_softmax(x), axis=1, keepdims=True)

print(modified_softmax([[1, 1], [2, 3]]))

from numpy import mean

l = [

    87.8,

    92.5,

    97.0,

    91.6,

    97.6,

    93.8,

    55.2,

    76.3

]
print(mean(l))
