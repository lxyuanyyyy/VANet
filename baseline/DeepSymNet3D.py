import torch.nn as nn
import torch
import torch.nn.functional as F


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception3D(nn.Module):
    def __init__(self, in_channels,
                 ch1x1x1=64,
                 ch3x3x3red=32, ch3x3x3=64,
                 ch5x5x5red=32, ch5x5x5=64,
                 pool_proj=64):
        super(Inception3D, self).__init__()
        self.branch1 = BasicConv3d(in_channels, ch1x1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv3d(in_channels, ch3x3x3red, kernel_size=1),
            BasicConv3d(ch3x3x3red, ch3x3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv3d(in_channels, ch5x5x5red, kernel_size=1),
            # 在官方的实现中，其实是3x3的kernel并不是5x5，这里我也懒得改了，具体可以参考下面的issue
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            BasicConv3d(ch5x5x5red, ch5x5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            BasicConv3d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class DeepSymNet3D(nn.Module):
    def __init__(self, num_classes=2, in_ch=1):
        super(DeepSymNet3D, self).__init__()

        self.conv1 = BasicConv3d(in_channels=in_ch, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool3d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv3d(64, 64, kernel_size=1)
        self.conv3 = BasicConv3d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool3d(3, stride=2, ceil_mode=True)

        # 前四个inception是共享权重部分
        self.inception1a = Inception3D(in_channels=192)
        self.inception1b = Inception3D(in_channels=256)
        self.maxpool3 = nn.MaxPool3d(3, stride=2, ceil_mode=True)
        # 14x14x480
        self.inception2a = Inception3D(in_channels=256)
        self.inception2b = Inception3D(in_channels=256)
        self.maxpool4 = nn.MaxPool3d(3, stride=2, ceil_mode=True)
        # 7x7x512

        self.conv1x1x1 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(256)
        self.inception3a = Inception3D(in_channels=256)
        self.inception3b = Inception3D(in_channels=256)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ada_maxpool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, num_classes)

    def share_weight_forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception1a(x)
        x = self.inception1b(x)
        x = self.maxpool3(x)
        x = self.inception2a(x)
        x = self.inception2b(x)
        x = self.maxpool4(x)
        return x

    def forward(self, x):
        #在配准的图像中直接使用中轴线分割左右半脑
        #MNI图像大小为182x218x182
        #MNI图像大小为193x229x193
        x_left = x[:, :, :, :, 0:128] #left brain
        x_right = x[:, :, :, :, 128:256] #right brain

        x_right_flip = torch.flip(x_right, dims=[4])

        x_left = self.share_weight_forward(x_left)
        x_right = self.share_weight_forward(x_right_flip)

        # 求特征差
        l1_diff = x_left - x_right
        # 求绝对值
        l1_diff = torch.abs(l1_diff)
        # 对绝对值操作后输入-
        l1_diff = F.relu(self.bn1(self.conv1x1x1(l1_diff)))
        out = self.inception3a(l1_diff)
        out = self.inception3b(out)
        out = self.ada_maxpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)

        return out
