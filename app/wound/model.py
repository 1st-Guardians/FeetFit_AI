"""ImageNet-pretrained Swin-T encoder with a convolutional U-Net decoder."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import Swin_T_Weights, swin_t


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, cout), cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, cout), cout), nn.GELU())


class UpBlock(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.block = ConvBlock(cin+skip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class SwinUNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = backbone.features
        self.norms = nn.ModuleList([nn.LayerNorm(96), nn.LayerNorm(192), nn.LayerNorm(384), backbone.norm])
        self.shallow = ConvBlock(3, 16)
        self.half_resolution = nn.Sequential(nn.Conv2d(16,32,3,stride=2,padding=1,bias=False),
                                             nn.GroupNorm(8,32),nn.GELU())
        self.up3 = UpBlock(768,384,256)
        self.up2 = UpBlock(256,192,128)
        self.up1 = UpBlock(128,96,64)
        self.up0 = UpBlock(64,32,32)
        self.up_full = UpBlock(32,16,16)
        self.head = nn.Conv2d(16,1,1)
        nn.init.constant_(self.head.bias, -2.)

    def forward(self, x):
        full = self.shallow(x)
        half = self.half_resolution(full)
        levels = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i in [1,3,5,7]:
                levels.append(self.norms[len(levels)](x).permute(0,3,1,2).contiguous())
        x = self.up3(levels[3],levels[2])
        x = self.up2(x,levels[1])
        x = self.up1(x,levels[0])
        x = self.up0(x,half)
        return self.head(self.up_full(x,full))
