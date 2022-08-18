import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common.model import BACKBONES


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def make_res_layer(inplanes, planes, blocks, stride=1):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        nn.BatchNorm3d(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))

    return nn.Sequential(*layers)


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)

def _ASPPConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    asppconv = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )
    return asppconv

class ASPP(nn.Module):
    """
    ASPP module in `DeepLabV3, see also in <https://arxiv.org/abs/1706.05587>` 
    """
    def __init__(self, in_channels, out_channels, output_stride=8):
        super(ASPP, self).__init__()

        if output_stride == 8:
            astrous_rates = [0, 2, 4, 8]
        else:
            raise Warning('Output stride must be 8 or 16!')

        # astrous spational pyramid pooling part
        self.conv1 = _ASPPConv(in_channels, out_channels, 1, 1)
        self.conv2 = _ASPPConv(in_channels, out_channels, 3, 1, padding=astrous_rates[1], dilation=astrous_rates[1])
        self.conv3 = _ASPPConv(in_channels, out_channels, 3, 1, padding=astrous_rates[2], dilation=astrous_rates[2])
        self.conv4 = _ASPPConv(in_channels, out_channels, 3, 1, padding=astrous_rates[3], dilation=astrous_rates[3])

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Conv3d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, input):
        input1 = self.conv1(input)
        input2 = self.conv2(input)
        input3 = self.conv3(input)
        input4 = self.conv4(input)
        
        input5 = F.interpolate(self.pool(input), size=input4.size()[2:], mode='trilinear', align_corners=False)
        output = torch.cat((input1, input2, input3, input4, input5), dim=1)
        output = self.bottleneck(output)
        return output


class ResUnet(nn.Module):

    def __init__(self, in_ch, channels=16, blocks=3, use_aspp=False, is_aux=False):
        super(ResUnet, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride=1, kernel_size=3)
        self.layer0 = make_res_layer(channels, channels * 2, blocks, stride=2)
        self.layer1 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer2 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 8, channels * 16, blocks, stride=2)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 24, channels * 8)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 12, channels * 4)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 6, channels * 2)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv8 = DoubleConv(channels * 3, channels * 1)

        self.aspp = ASPP(channels * 16, channels * 16)
        self.is_aux = is_aux
        self.use_aspp = use_aspp

    def forward(self, input):
        c0 = self.in_conv(input) 
        c1 = self.layer0(c0) 
        c2 = self.layer1(c1) 
        c3 = self.layer2(c2) 
        c4 = self.layer3(c3) 

        if self.use_aspp:
            c4_ap = self.aspp(c4) 

        up_5 = self.up5(c4_ap)
        merge5 = torch.cat([up_5, c3], dim=1)
        c5 = self.conv5(merge5) 
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2], dim=1)
        c6 = self.conv6(merge6) 
        up_7 = self.up7(c6) 
        merge7 = torch.cat([up_7, c1], dim=1)
        c7 = self.conv7(merge7) 
        up_8 = self.up8(c7) 
        merge8 = torch.cat([up_8, c0], dim=1)
        c8 = self.conv8(merge8)
        if self.is_aux:
            return [c8, c7, c6, c5]
        else:
            return c8


if __name__ == '__main__':
    model = ResUnet(3, 32)
    print('#generator parameters:', sum(param.numel() for param in model.parameters()))
    # classical ResUnet (downsample=16, channel=16) params: 3796816
    # classical ResUnet (downsample=16, channels=32) params: 15176864
    # MW ResUnet (downsample=32, channels=16) params: 15176864
