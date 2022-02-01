import torch
from torch import nn


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2))
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.block1 = ResBlock(64, 64, 1)
        self.block2 = ResBlock(64, 128, 2)
        self.block3 = ResBlock(128, 256, 2)
        self.block4 = ResBlock(256, 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = self.avgpool(out)
        out = torch.reshape(out, (-1, 512))
        out = self.fc(out)
        out = self.sigmoid(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.convx = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride))
        self.bnx = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        x = self.convx(x)
        x = self.bnx(x)
        out += x
        out = self.relu(out)

        return out
