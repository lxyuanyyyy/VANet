#relation-aware global attention
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
from torch.nn import functional as F


import math
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch as th
from torch.autograd import Variable
import pdb


# ===================
#     RGA Module
# ===================

class RGA_Module(nn.Module):
    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True, \
                 cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(RGA_Module, self).__init__()

        #in_spatial参数含义 特征图像素个数 h*w

        self.in_channel = in_channel
        self.in_spatial = in_spatial

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        print('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio

        # Embedding functions for original features
        if self.use_spatial:
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.gx_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

        # Embedding functions for relation features
        if self.use_spatial:
            self.gg_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
        if self.use_channel:
            self.gg_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

        # Networks for learning attention weights
        if self.use_spatial:
            num_channel_s = 1 + self.inter_spatial
            self.W_spatial = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_s // down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_s // down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
        if self.use_channel:
            num_channel_c = 1 + self.inter_channel
            self.W_channel = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c // down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_c // down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_c // down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )

        # Embedding functions for modeling relations
        if self.use_spatial:
            self.theta_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.theta_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            self.phi_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

    def forward(self, x):
        b, c, h, w = x.size()

        if self.use_spatial:
            # spatial attention
            theta_xs = self.theta_spatial(x)
            phi_xs = self.phi_spatial(x)
            theta_xs = theta_xs.view(b, self.inter_channel, -1)
            theta_xs = theta_xs.permute(0, 2, 1)
            phi_xs = phi_xs.view(b, self.inter_channel, -1)
            Gs = torch.matmul(theta_xs, phi_xs)
            Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)
            Gs_out = Gs.view(b, h * w, h, w)
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)
            Gs_joint = self.gg_spatial(Gs_joint)

            g_xs = self.gx_spatial(x)
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)
            ys = torch.cat((g_xs, Gs_joint), 1)

            W_ys = self.W_spatial(ys)
            #不使用通道注意力则直接返回
            if not self.use_channel:
                spatial_attn_mask=torch.sigmoid(W_ys.expand_as(x))
                out = spatial_attn_mask * x
                return out,spatial_attn_mask
            #使用通道注意力则保存当前注意力后的tensor
            else:
                x = torch.sigmoid(W_ys.expand_as(x)) * x

        if self.use_channel:
            # channel attention
            xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
            theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
            phi_xc = self.phi_channel(xc).squeeze(-1)
            Gc = torch.matmul(theta_xc, phi_xc)
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
            Gc_out = Gc.unsqueeze(-1)
            Gc_joint = torch.cat((Gc_in, Gc_out), 1)
            Gc_joint = self.gg_channel(Gc_joint)

            g_xc = self.gx_channel(xc)
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)
            yc = torch.cat((g_xc, Gc_joint), 1)

            W_yc = self.W_channel(yc).transpose(1, 2)
            out = torch.sigmoid(W_yc) * x

            return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_fc(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RGA_Branch(nn.Module):
    def __init__(self, pretrained=False, last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3],
                 spa_on=True, cha_on=True, s_ratio=8, c_ratio=8, d_ratio=8, height=256, width=128,
                 model_path=''):
        super(RGA_Branch, self).__init__()

        self.in_channels = 64

        # Networks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        # RGA Modules
        self.rga_att1 = RGA_Module(256, (height // 4) * (width // 4), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
        self.rga_att2 = RGA_Module(512, (height // 8) * (width // 8), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
        self.rga_att3 = RGA_Module(1024, (height // 16) * (width // 16), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
        self.rga_att4 = RGA_Module(2048, (height // 16) * (width // 16), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)

        # Load the pre-trained model weights
        if pretrained:
            self.load_specific_param(self.conv1.state_dict(), 'conv1', model_path)
            self.load_specific_param(self.bn1.state_dict(), 'bn1', model_path)
            self.load_partial_param(self.layer1.state_dict(), 1, model_path)
            self.load_partial_param(self.layer2.state_dict(), 2, model_path)
            self.load_partial_param(self.layer3.state_dict(), 3, model_path)
            self.load_partial_param(self.layer4.state_dict(), 4, model_path)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def load_partial_param(self, state_dict, model_index, model_path):
        param_dict = torch.load(model_path)
        for i in state_dict:
            key = 'layer{}.'.format(model_index) + i
            state_dict[i].copy_(param_dict[key])
        del param_dict

    def load_specific_param(self, state_dict, param_name, model_path):
        param_dict = torch.load(model_path)
        for i in state_dict:
            key = param_name + '.' + i
            state_dict[i].copy_(param_dict[key])
        del param_dict

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.rga_att1(x)

        x = self.layer2(x)
        x = self.rga_att2(x)

        x = self.layer3(x)
        x = self.rga_att3(x)

        x = self.layer4(x)
        x = self.rga_att4(x)

        return x


#常用参数设置