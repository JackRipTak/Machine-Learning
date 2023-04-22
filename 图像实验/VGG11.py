import mindspore.nn as nn


class VGG11(nn.Cell):
    """
    网络结构
    """

    def __init__(self, num_class=10, num_channel=3):
        super(VGG11, self).__init__()
        # 定义所需要的运算

    def conv_block(self, in_channel, out_channel):
        # 定义卷积块的结构

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        return x