import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from .CBAM import *
from .RGA import *
from .triplet_attn import *
from .coordinate_attn import *

#返回空间注意力mask
class CBAM_spatial(nn.Module):
    #先channel 再返回 空间注意力mask
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_spatial, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialScale = SpatialAttn()

    def forward(self, x):
        #通道注意力
        x_out = self.ChannelGate(x)

        spatial_attn_mask=self.SpatialScale(x_out)

        return spatial_attn_mask

class RGA_spatial(nn.Module):
    def __init__(self,in_channel,in_spatial):
        super(RGA_spatial, self).__init__()

        self.channel_attn_only=RGA_Module(in_channel,in_spatial,use_spatial=False,use_channel=True)
        self.spatial_attn_only=RGA_Module(in_channel,in_spatial,use_spatial=True,use_channel=False)

    def forward(self,x):
        x=self.channel_attn_only(x)
        x,mask=self.spatial_attn_only(x)

        return mask

class CoorAttn_spatial(nn.Module):
    pass