import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_

from pcdet.models.model_utils.basic_block_2d import build_block


class TeacherAlignLayer(nn.Module):
    def __init__(self, in_channels, out_c1=256, out_c2=32):
        super().__init__()
        self.out_c1=out_c1
        self.out_c2 = out_c2
        self.conv256 = nn.Sequential(
            nn.Conv2d(in_channels, out_c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True)
        )
        self.bn256 = nn.BatchNorm2d(out_c1)
        self.conv32 = nn.Sequential(
            nn.Conv2d(out_c1, out_c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c2),
            nn.ReLU(inplace=True)
        )
        self.bn32 = nn.BatchNorm2d(out_c2)
        self.relu = nn.ReLU(inplace=True)

        # self._init_weights()
        self.channel_select_init(self.conv32)
        self.channel_select_init(self.conv256)

    # def _init_weights(self):
    #     nn.init.orthogonal_(self.conv256.weight)
    #     nn.init.orthogonal_(self.conv32.weight)
    def channel_select_init(self,conv):
        with torch.no_grad():
            conv.weight.zero_()
            out_c, in_c, _, _ =conv.weight.shape
            step = in_c /out_c
            for i in range(out_c):
                start = int(i*step)
                end = int((i+1)*step)
                k = max(end-start, 1)
                conv.weight[i,start:end,0,0] = 1.0/k
    def channel_dowmsample(self,x):
        B,C,H,W = x.shape
        step = C// self.out_c1
        idx = torch.arange(0,C,step,device=x.device)[:self.out_c1]
        return x[:,idx,:,:]
    def forward(self, feat):
        # residual256 = self.channel_dowmsample(feat)
        feat256 = self.relu(self.bn256(self.conv256(feat)))
        feat32 = self.relu(self.bn32(self.conv32(feat256)))
        
        return feat256, feat32

class StudentAlignLayer(nn.Module):
    def __init__(self, in_channel=256, out_channel=32):
        super().__init__()
        self.conv32 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.bn32 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        # self._init_weights()
        self.channel_select_init(self.conv32)

    # def _init_weights(self):
    #     nn.init.orthogonal_(self.conv32.weight)
    def channel_select_init(self,conv):
        with torch.no_grad():
            conv.weight.zero_()
            out_c, in_c, _, _ =conv.weight.shape
            step = in_c /out_c
            for i in range(out_c):
                j = int(i*step)
                conv.weight[i,j,0,0] = 1.0

    def forward(self, feat):
        feat256 = feat
        feat32 = self.relu(self.bn32(self.conv32(feat)))
        
        return feat256, feat32

