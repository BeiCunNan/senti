import torch.cuda
from pynvml import *


def show_gpu(simlpe=True):
    # 初始化
    nvmlInit()
    # 获取GPU个数
    deviceCount = nvmlDeviceGetCount()
    total_memory = 0
    total_free = 0
    total_used = 0
    gpu_name = ""
    gpu_num = deviceCount

    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_name = nvmlDeviceGetName(handle).decode('utf-8')
        # 查看型号、显存、温度、电源
        if not simlpe:
            print("[ GPU{}: {}".format(i, gpu_name), end="    ")
            print("总共显存: {}G".format((info.total // 1048576) / 1024), end="    ")
            print("空余显存: {}G".format((info.free // 1048576) / 1024), end="    ")
            print("已用显存: {}G".format((info.used // 1048576) / 1024), end="    ")
            print("显存占用率: {}%".format(info.used / info.total), end="    ")
            print("运行温度: {}摄氏度 ]".format(nvmlDeviceGetTemperature(handle, 0)))

        total_memory += (info.total // 1048576) / 1024
        total_free += (info.free // 1048576) / 1024
        total_used += (info.used // 1048576) / 1024

    print("显卡名称：[{}]，显卡数量：[{}]，总共显存；[{}G]，空余显存：[{}G]，已用显存：[{}G]，显存占用率：[{}%]。".format(gpu_name,
                                                                                                              gpu_num,
                                                                                                              total_memory,
                                                                                                              total_free,
                                                                                                              total_used,
                                                                                                              (
                                                                                                                          total_used / total_memory)))

    # 关闭管理工具
    nvmlShutdown()


def use_gpu(used_percentage=0.75):
    '''
    不使用显存占用率高于used_percentage的gpu
    :param used_percentage:
    :return:
    '''
    nvmlInit()
    gpu_num = nvmlDeviceGetCount()
    out = ""
    for i in range(gpu_num):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        used_percentage_real = info.used / info.total
        if out == "":
            if used_percentage_real < used_percentage:
                out += str(i)
        else:
            if used_percentage_real < used_percentage:
                out += "," + str(i)
    nvmlShutdown()
    return out


show_gpu(False)
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu(0.5)  # 选择使用训练的GPU

