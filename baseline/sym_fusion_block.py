import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .CBAM import *
from .RGA import *
from .triplet_attn import *
from .coordinate_attn import *

class l1_block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(l1_block, self).__init__()
        pass



#仅利用特征翻转差
class get_difference(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x1卷积核
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        # BN层
        self.bn1 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x_flip = torch.flip(x, dims=[3])
        x = x - x_flip
        out = F.relu(self.bn1(self.conv1(x)))
        return x

# 特征翻转差的残差连接
class sym_fusion_res_block(nn.Module):

    def __init__(self, in_channels, out_channels, drop_path=0.1,abs=True):
        super().__init__()
        # 1x1卷积核
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        # BN层
        self.bn1 = nn.BatchNorm2d(in_channels)
        # self.l1_flip=flip_l1_block(in_channels,out_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.abs=abs

    def forward(self, x):
        x_flip = torch.flip(x, dims=[3])
        out = x - x_flip
        if self.abs:
            out=torch.abs(out)
        out = F.relu(self.bn1(self.conv1(out)))
        return x + self.drop_path(out)

# attention 残差连接
class sym_fusion_attn_block(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path=0.3, abs=True):
        super().__init__()
        # 1x1卷积核
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        # BN层
        self.bn1 = nn.BatchNorm2d(in_channels)
        # self.l1_flip=flip_l1_block(in_channels,out_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.abs=abs

    def forward(self, x):
        x_flip = torch.flip(x, dims=[3])
        out = x - x_flip
        if self.abs:
            out=torch.abs(out)
        out = F.relu(self.bn1(self.conv1(out)))
        out = torch.sigmoid(out)
        out = out * x
        return x+self.drop_path(out)



# attention 模块
class SE_block(nn.Module):
    def __init__(self):
        super().__init__()
        pass
class res_CAM(nn.Module):
    def __init__(self):
        super(res_CAM, self).__init__()
        pass
class res_SAM(nn.Module):
    def __init__(self):
        super(res_SAM, self).__init__()
        pass


class res_CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, no_spatial=False, drop_path=0.3,abs=False):
        super().__init__()
        self.cbam = CBAM(gate_channels=gate_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 1x1卷积核
        self.conv1 = nn.Conv2d(in_channels=gate_channels,
                               out_channels=gate_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        # BN层
        self.bn1 = nn.BatchNorm2d(gate_channels)
        # self.l1_flip=flip_l1_block(in_channels,out_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x_flip = torch.flip(x, dims=[3])
        out = x - x_flip
        out = self.cbam(out)
        out = F.relu(self.bn1(self.conv1(out)))
        return x + self.drop_path(out)

#对特征差做空间注意力和通道注意力
class res_RGA(nn.Module):
    def __init__(self,in_channels=64,drop_path=0.3,feature_map_size=256,down_sample_ratio=1, abs=False,use_channel=False):
        super(res_RGA, self).__init__()

        self.abs=abs
        #self.down_sample=nn.AvgPool2d(kernel_size=down_sample_ratio,stride=down_sample_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 1x1卷积核
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        # BN层
        self.bn1 = nn.BatchNorm2d(in_channels)
        # self.l1_flip=flip_l1_block(in_channels,out_channels)
        #dot_sum=(feature_map_size//down_sample_ratio)*(feature_map_size//down_sample_ratio)
        #默认仅使用空间注意力模块
        self.attn = RGA_Module(in_channel=in_channels,in_spatial=feature_map_size,use_channel=use_channel)

    def forward(self,x):
        x_flip = torch.flip(x, dims=[3])
        out = x - x_flip

        if self.abs:
            out = torch.abs(out)

        out = self.attn(out)
        out = F.relu(self.bn1(self.conv1(out)))
        #残差连接
        return x + self.drop_path(out)

#通过下采样使每层特征的空间注意力权重共享
class shared_RGA_spatial(nn.Module):
    def __init__(self,in_channels=64,drop_path=0.3,feature_map_size=256):
        super(shared_RGA_spatial, self).__init__()
        #feature map size为特征图的 h*w
        self.spatial_attn=RGA_Module(in_channel=in_channels,
                                     in_spatial=feature_map_size,
                                     use_spatial=True,use_channel=False)



#对特征差做空间注意力和通道注意力
#coordinate
class res_CA(nn.Module):
    def __init__(self,in_channels=64,drop_path=0.2, abs=False):
        super(res_CA, self).__init__()

        self.abs=abs
        #self.down_sample=nn.AvgPool2d(kernel_size=down_sample_ratio,stride=down_sample_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 1x1卷积核
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        # BN层
        self.bn1 = nn.BatchNorm2d(in_channels)
        # self.l1_flip=flip_l1_block(in_channels,out_channels)
        #dot_sum=(feature_map_size//down_sample_ratio)*(feature_map_size//down_sample_ratio)
        #默认仅使用空间注意力模块
        self.attn = CoordAtt(inp=in_channels,oup=in_channels)
    def forward(self,x):
        x_flip = torch.flip(x, dims=[3])
        out = x - x_flip

        if self.abs:
            out = torch.abs(out)

        out = self.attn(out)
        out = F.relu(self.bn1(self.conv1(out)))
        #残差连接
        return x + self.drop_path(out)

#对特征差做空间注意力和通道注意力
#triplet attn
class res_TA(nn.Module):
    def __init__(self,in_channels=64,drop_path=0.2,feature_map_size=256,down_sample_ratio=1, abs=False,use_channel=False):
        super(res_TA, self).__init__()

        self.abs=abs
        #self.down_sample=nn.AvgPool2d(kernel_size=down_sample_ratio,stride=down_sample_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 1x1卷积核
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        # BN层
        self.bn1 = nn.BatchNorm2d(in_channels)
        # self.l1_flip=flip_l1_block(in_channels,out_channels)
        #dot_sum=(feature_map_size//down_sample_ratio)*(feature_map_size//down_sample_ratio)
        #默认仅使用空间注意力模块
        self.attn = TripletAttention()

    def forward(self,x):
        x_flip = torch.flip(x, dims=[3])
        out = x - x_flip

        if self.abs:
            out = torch.abs(out)

        out = self.attn(out)
        out = F.relu(self.bn1(self.conv1(out)))
        #残差连接
        return x + self.drop_path(out)