import json
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from matplotlib import gridspec
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.ticker as ticker
from data import MyDataset, my_collate

def draw(index):
    def modified_softmax(x):
        def modified_softmax(x):
            exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_values / (np.sum(exp_values, axis=1, keepdims=True) + 1e-6)

        def inverse_softmax(x):
            return 1 - modified_softmax(x)

        return inverse_softmax(x) / np.sum(inverse_softmax(x), axis=1, keepdims=True)


    def modified_sigmoid(x):
        def modified_sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def inverse_sigmoid(x):
            return 1 - modified_sigmoid(x)

        return inverse_sigmoid(x) / np.sum(inverse_sigmoid(x), axis=1, keepdims=True)


    def max_min(x):
        def gap(x):
            return np.max(x, axis=1, keepdims=True) - x

        return gap(x) / (np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True))


    def modified_softmax1(x):
        exp_values = np.exp(np.max(x, axis=1, keepdims=True) - x)
        return exp_values / (np.sum(exp_values, axis=1, keepdims=True) + 1e-6)


    SENTENCE = 'He is scared stiff of it'
    MRC_SENTENCE = SENTENCE + ' [SEP] What class in { positive , negative , other } does this sentence have ?'
    PL_SENTENCE = SENTENCE + ' [SEP] This idiom is [MASK] .'

    context_split = SENTENCE.split(' ')
    mrc_split = MRC_SENTENCE.split(' ')
    pl_split = PL_SENTENCE.split(' ')

    context_length = len(context_split)
    mrc_length = len(mrc_split)
    pl_length = len(pl_split)

    "if they can get a foothold in europe they will have the chance to be bigger and better"
    "如果他们能在欧洲站稳脚跟，他们将有机会变得更大更好。"
    "soon it was dark and the christmas evening was in full swing"
    "很快天黑了，圣诞之夜如火如荼"
    "he was a strong healthy lad and as pleased as punch to be working with dad"
    "他是一个强壮健康的小伙子，很高兴能和爸爸一起工作"
    "other critics who could not get worked up about the play also admired the acting"
    "其他对这部剧不满意的评论家也对演技赞不绝口。"
    "not since the surprise smash hit of the year cast a rosy glow over shores vehicle"
    "自从当年出人意料的热门歌曲在海岸车辆上投下玫瑰色的光芒以来"
    "i am delighted that they all passed with flying colours"
    "我很高兴他们都以优异的成绩通过了"
    "come on girl take it easy"
    "来吧女孩，放轻松一点"

    "moreover death is just the tip of the iceberg"
    "此外，死亡只是冰山一角"
    "the market is down in the dumps"
    "市场在倾销中下跌"
    "its simply asking for trouble"
    "它只是自找麻烦"
    "one false move and youre dead"
    "一个错误的举动，你就死了"
    "obviously these two were at loggerheads"
    "显然这两个人争执不下"
    "hes scared stiff of it"
    "他吓得僵硬"
    "oh pack it in this is serious"
    "哦，把它包装成这个很严重"

    print("Getting the model")
    model = torch.load('./save/model.pkl')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model.eval()

    print("Data processing")
    case_data = json.load(open(os.path.join('./data', 'IE_Case.json'), 'r', encoding='utf-8'))
    label_dict = {'positive': 0, 'negative': 1, 'other': 2}
    caseSet = MyDataset(case_data, label_dict, 'idiom')
    collate_fn = partial(my_collate, tokenizer=tokenizer)
    case_dataloader = DataLoader(caseSet, 1, shuffle=False, num_workers=0, collate_fn=collate_fn,
                                 pin_memory=True)
    print("Getting the result")
    for mrc_inputs, targets, text_inputs, mask_inputs, mask_index in tqdm(case_dataloader, disable=False,
                                                                          ascii=' >='):
        mrc_inputs = {k: v.to('cuda') for k, v in mrc_inputs.items()}
        text_inputs = {k: v.to('cuda') for k, v in text_inputs.items()}
        mask_inputs = {k: v.to('cuda') for k, v in mask_inputs.items()}
        targets = targets.to('cuda')
        predicts, aTSA, aFSA, mrc_tokens, mrc_CLS, bTSA, bFSA, context_tokens, text_CLS, cTSA, cFSA, pl_tokens, MASK = model(
            mrc_inputs, text_inputs, mask_inputs, mask_index)
        print(predicts)

    print("Calculating l2")
    context_CLS = text_CLS.repeat(context_length, 1)
    tvsa_l2 = torch.nn.functional.pairwise_distance(bTSA[0, 1:context_length + 1, :], context_CLS, p=2,
                                                    eps=1e-06)
    fvsa_l2 = torch.nn.functional.pairwise_distance(bFSA[0, 1:context_length + 1, :], context_CLS, p=2,
                                                    eps=1e-06)
    context_tvsa = torch.cat((tvsa_l2, fvsa_l2), dim=0).cpu().detach().numpy()
    context_tvsa = numpy.reshape(context_tvsa, (2, context_length))

    context_tvsa = numpy.reshape(context_tvsa, (2, context_length))
    context_l2 = max_min(context_tvsa)
    # max_values = np.amax(context_tvsa, axis=1)
    # max_values_expanded = np.expand_dims(max_values, axis=1)
    # max_values_expanded = np.repeat(max_values_expanded, context_tvsa.shape[1], axis=1)
    # context_l2 = max_values_expanded - context_tvsa

    mrc_CLS = text_CLS.repeat(mrc_length, 1)
    mrc_tvsa = torch.nn.functional.pairwise_distance(aTSA[0, 1:mrc_length + 1, :], mrc_CLS, p=2,
                                                     eps=1e-06)
    mrc_fvsa = torch.nn.functional.pairwise_distance(aFSA[0, 1:mrc_length + 1, :], mrc_CLS, p=2,
                                                     eps=1e-06)
    mrc_tvsa = torch.cat((mrc_tvsa, mrc_fvsa), dim=0).cpu().detach().numpy()
    mrc_tvsa = numpy.reshape(mrc_tvsa, (2, mrc_length))
    mrc_l2 = max_min(mrc_tvsa)
    # max_values = np.amax(mrc_tvsa, axis=1)
    # max_values_expanded = np.expand_dims(max_values, axis=1)
    # max_values_expanded = np.repeat(max_values_expanded, mrc_tvsa.shape[1], axis=1)
    # mrc_l2 = max_values_expanded - mrc_tvsa

    pl_MASK = mrc_CLS.repeat(pl_length, 1)
    pl_tvsa = torch.nn.functional.pairwise_distance(cTSA[0, 1:pl_length + 1, :], MASK, p=2, eps=1e-06)
    pl_fvsa = torch.nn.functional.pairwise_distance(cFSA[0, 1:pl_length + 1, :], MASK, p=2, eps=1e-06)
    pl_tvsa = torch.cat((pl_tvsa, pl_fvsa), dim=0).cpu().detach().numpy()
    pl_tvsa = numpy.reshape(pl_tvsa, (2, pl_length))
    pl_l2 = max_min(pl_tvsa)
    # max_values = np.amax(pl_tvsa, axis=1)
    # max_values_expanded = np.expand_dims(max_values, axis=1)
    # max_values_expanded = np.repeat(max_values_expanded, pl_tvsa.shape[1], axis=1)
    # pl_l2 = max_values_expanded - pl_tvsa

    print("Drawing")
    choice = 3



    # 创建网格布局
    fig = plt.figure()
    fig.set_size_inches(12, 6)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], width_ratios=[1])

    # 绘制第一个热力图
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(mrc_l2, interpolation='none', cmap='Blues', extent=[0, mrc_length, 0, 2])
    ax1.set_xticks(np.arange(mrc_length))
    ax1.xaxis.set_major_locator(ticker.FixedLocator(np.arange(mrc_length) +0.5))
    ax1.xaxis.set_major_formatter(ticker.FixedFormatter(mrc_split))
    ax1.set_xticklabels(
        mrc_split,
        rotation=40, rotation_mode="anchor", ha="right")

    ax1.set_yticks(np.arange(2))
    ax1.yaxis.set_major_locator(ticker.FixedLocator(np.arange(2) + 0.5))
    ax1.set_yticklabels(['FVSA', 'TVSA'])
    ax1.set_title('MRC-IE')

    # 绘制第二个热力图
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(context_l2, interpolation='none', cmap='Blues', extent=[0, context_length, 0, 2])
    ax2.set_xticks(np.arange(context_length))
    ax2.xaxis.set_major_locator(ticker.FixedLocator(np.arange(context_length) +0.5))
    ax2.xaxis.set_major_formatter(ticker.FixedFormatter(context_split))
    ax2.set_xticklabels(
        context_split,
        rotation=40, rotation_mode="anchor",ha="right")
    ax2.set_yticks(np.arange(2))
    ax2.yaxis.set_major_locator(ticker.FixedLocator(np.arange(2) + 0.5))

    ax2.set_yticklabels(['FVSA', 'TVSA'])
    ax2.set_title('Context-IE')

    # 绘制第三个热力图
    ax3 = fig.add_subplot(gs[2, 0])
    im3 = ax3.imshow(pl_l2, interpolation='none', cmap='Blues', extent=[0, pl_length, 0, 2])
    ax3.set_xticks(np.arange(pl_length))
    ax3.xaxis.set_major_locator(ticker.FixedLocator(np.arange(pl_length) +0.5))
    ax3.xaxis.set_major_formatter(ticker.FixedFormatter(pl_split))
    ax3.set_xticklabels(
        pl_split,
        rotation=40, rotation_mode="anchor", ha="right")

    ax3.set_yticks(np.arange(2))
    ax3.yaxis.set_major_locator(ticker.FixedLocator(np.arange(2) + 0.5))
    ax3.set_yticklabels(['FVSA', 'TVSA'])
    ax3.set_title('PL-IE')

    # 调整子图之间的间距
    # plt.tight_layout()
    plt.subplots_adjust(top=0.8, bottom=0.1, hspace=1.2, wspace=0)


    gs.update(wspace=0)
    gs.tight_layout(fig, rect=[0, 0, 1, 1])

    ax1_pos = ax1.get_position()
    ax1.set_position([0.06, ax1_pos.y0, ax1_pos.width, ax1_pos.height])

    ax2_pos = ax2.get_position()
    ax2.set_position([0.06, ax2_pos.y0, ax2_pos.width, ax2_pos.height])

    ax3_pos = ax3.get_position()
    ax3.set_position([0.06, ax3_pos.y0, ax3_pos.width, ax3_pos.height])

    # 显示图表

    cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3])

    plt.savefig('./pic/'+str(index)+'.png')

    # plt.show()


# def draw():
#     fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#     heatmaps = [mrc_l2, context_l2, pl_l2]
#     titles = ['MRC-IE', 'Context-IE', 'PL-IE']
#     for i, ax in enumerate(axs):
#         ax.imshow(heatmaps[i], interpolation='none', cmap='Blues')
#         ax.set_title(titles[i])
#         ax.set_xticks(np.arange(context_length))
#         ax.set_xticklabels(context_split, rotation=45, rotation_mode="anchor", ha="right")
#         ax.set_yticks(np.arange(2))
#         ax.set_yticklabels(['tvsa', 'fvsa'])
#     plt.suptitle('Merged Heatmaps')
#     plt.tight_layout()
#     #plt.colorbar()
#     plt.show()

# draw(99)

# if choice == 1:
#     plt.imshow(mrc_l2, interpolation='none', cmap='Blues')
#     plt.xticks(np.arange(mrc_length), labels=mrc_split, rotation=45, rotation_mode="anchor", ha="right")
#     plt.yticks(np.arange(2), labels=['tvsa', 'fvsa'])
#     plt.suptitle('MRC-IE')
# elif choice == 2:
#     plt.imshow(context_l2, interpolation='none', cmap='Blues')
#     plt.xticks(np.arange(context_length), labels=context_split, rotation=45, rotation_mode="anchor", ha="right")
#     plt.yticks(np.arange(2), labels=['tvsa', 'fvsa'])
#     plt.suptitle('Context-IE')
# else:
#     plt.imshow(pl_l2, interpolation='none', cmap='Blues')
#     plt.xticks(np.arange(pl_length), labels=pl_split, rotation=45, rotation_mode="anchor", ha="right")
#     plt.yticks(np.arange(2), labels=['tvsa', 'fvsa'])
#     plt.suptitle('PL-IE')

# plt.show()
