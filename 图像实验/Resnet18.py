import mindspore.nn as nn


class ResBlockDS(nn.Cell):


class ResBlock(nn.Cell):


class ResNet18(nn.Cell):
    def __init__(self, class_num=10, in_channel=3):
        super(ResNet18, self).__init__()
        self.conv = nn.Conv2d(in_channel, 64, 7, stride=2, padding=3, pad_mode='pad')
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ResBlock1 = ResBlock(64, 64)
        self.ResBlock2 = ResBlock(64, 64)
        self.ResBlock3 = ResBlockDS(64, 128)
        self.ResBlock4 = ResBlock(128, 128)
        self.ResBlock5 = ResBlockDS(128, 256)
        self.ResBlock6 = ResBlock(256, 256)
        self.ResBlock7 = ResBlockDS(256, 512)
        self.ResBlock8 = ResBlock(512, 512)
        self.LastProcess = nn.SequentialCell([
            nn.AvgPool2d(1),
            nn.Flatten(),
            nn.Dense(512, 1024),
            nn.ReLU(),
            nn.Dense(1024, 512),
            nn.ReLU(),
            nn.Dropout(keep_prob=0.5),
            nn.Dense(512, class_num),
        ])

    def construct(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)
        x = self.ResBlock5(x)
        x = self.ResBlock6(x)
        x = self.ResBlock7(x)
        x = self.ResBlock8(x)
        y = self.LastProcess(x)
        return y