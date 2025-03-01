import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class DESNET(nn.Module):
    def __init__(self):
        super(DESNET, self).__init__()

        self.norm = nn.InstanceNorm2d(1)

        self.block1 = nn.Sequential(
                                    Block(1, 6, stride=1),
                                    Block(6, 12, stride=2),
                                    Block(12, 24, stride=1),
                                    Block(24, 48, stride=2),
        )
        self.block2 = nn.Sequential(
                                    Block(48, 48, stride=1),
                                    Block(48, 96, stride=1),
        )
        self.block3 = nn.Sequential(
                                    Block(96, 128, stride=2),
                                    Block(128, 128, stride=1),
                                    Block(128, 128, 1, padding=0),
        )
        self.block4 = nn.Sequential(
                                    Block(128, 128, stride=2),
                                    Block(128, 128, stride=1),
                                    Block(128, 128, stride=1),
        )
        self.block5 = nn.Sequential(
                                    Block(128, 256, stride=2),
                                    Block(256, 256, stride=1),
                                    Block(256, 256, stride=1),
                                    Block(256, 128, 1, padding=0),
        )
        self.des = nn.Sequential(
                                    Block(128, 128, stride=1),
                                    Block(128, 128, stride=1),
                                    nn.Conv2d(128, 128, 1, padding=0)
        )

    def forward(self, x):
        # 通过VGG提取特征
        with torch.no_grad():
            x = self.norm(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')

        des = self.des( x3 + x4 + x5 ) #(batch_size, 128, H/8, W/8)
        return des
