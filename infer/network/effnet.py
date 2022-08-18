import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common.model import BACKBONES

# norm = nn.InstanceNorm3d
# norm = nn.BatchNorm3d


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _ConvINReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, p=0.2):
        super(_ConvINReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm3d(out_channels)
        self.drop = nn.Dropout3d(p=p, inplace=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.relu(x)

        return x


class _ConvIN3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(_ConvIN3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

class AnisotropicMaxPooling(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(12, 12, 12), is_dynamic_empty_cache=False):
        super(AnisotropicMaxPooling, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(4, 4, 4))
        self.pool3 = nn.MaxPool3d(kernel_size=(kernel_size[0], 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, kernel_size[1], 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 1, kernel_size[2]))

        inter_channel = in_channel // 4

        self.trans_layer = _ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_1 = _ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_2 = _ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv2_0 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_1 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_2 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_3 = _ConvIN3D(inter_channel, inter_channel, (1, 3, 3), stride=1, padding=(1, 0, 0))
        self.conv2_4 = _ConvIN3D(inter_channel, inter_channel, (3, 1, 3), stride=1, padding=(0, 1, 0))
        self.conv2_5 = _ConvIN3D(inter_channel, inter_channel, (3, 3, 1), stride=1, padding=(0, 0, 1))

        self.conv2_6 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv2_7 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv3 = _ConvIN3D(inter_channel*2, inter_channel, 1, stride=1, padding=0)
        self.score_layer = nn.Sequential(_ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2),
                                         nn.Conv3d(inter_channel, out_channel, 1, bias=False))

    def forward(self, x):
        size = x.size()[2:]
        x0 = self.trans_layer(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        x1 = self.conv1_1(x0)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), size, mode='trilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), size, mode='trilinear', align_corners=True)
        out1 = self.conv2_6(F.relu(x2_1 + x2_2 + x2_3, inplace=True))
        if self.is_dynamic_empty_cache:
            del x1, x2_1, x2_2, x2_3
            torch.cuda.empty_cache()

        x2 = self.conv1_2(x0)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), size, mode='trilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), size, mode='trilinear', align_corners=True)
        x2_6 = F.interpolate(self.conv2_5(self.pool5(x2)), size, mode='trilinear', align_corners=True)
        out2 = self.conv2_7(F.relu(x2_4 + x2_5 + x2_6, inplace=True))
        if self.is_dynamic_empty_cache:
            del x2, x2_4, x2_5, x2_6
            torch.cuda.empty_cache()

        out = self.conv3(torch.cat([out1, out2], dim=1))
        out = F.relu(x0 + out, inplace=True)
        if self.is_dynamic_empty_cache:
            del x0, out1, out2
            torch.cuda.empty_cache()

        out = self.score_layer(out)

        return out

class AnisotropicAvgPooling(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(12, 12, 12), is_dynamic_empty_cache=False):
        super(AnisotropicAvgPooling, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.pool1 = nn.AvgPool3d(kernel_size=(2, 2, 2))
        self.pool2 = nn.AvgPool3d(kernel_size=(4, 4, 4))
        self.pool3 = nn.AvgPool3d(kernel_size=(1, kernel_size[1], kernel_size[2]))
        self.pool4 = nn.AvgPool3d(kernel_size=(kernel_size[0], 1, kernel_size[2]))
        self.pool5 = nn.AvgPool3d(kernel_size=(kernel_size[0], kernel_size[1], 1))

        inter_channel = in_channel // 4

        self.trans_layer = _ConvINReLU3D(in_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_1 = _ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv1_2 = _ConvINReLU3D(inter_channel, inter_channel, 1, stride=1, padding=0, p=0.2)
        self.conv2_0 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_1 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_2 = _ConvIN3D(inter_channel, inter_channel, 3, stride=1, padding=1)
        self.conv2_3 = _ConvIN3D(inter_channel, inter_channel, (3, 1, 1), stride=1, padding=(1, 0, 0))
        self.conv2_4 = _ConvIN3D(inter_channel, inter_channel, (1, 3, 1), stride=1, padding=(0, 1, 0))
        self.conv2_5 = _ConvIN3D(inter_channel, inter_channel, (1, 1, 3), stride=1, padding=(0, 0, 1))

        self.conv2_6 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv2_7 = _ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2)
        self.conv3 = _ConvIN3D(inter_channel*2, inter_channel, 1, stride=1, padding=0)
        self.score_layer = nn.Sequential(_ConvINReLU3D(inter_channel, inter_channel, 3, stride=1, padding=1, p=0.2),
                                         nn.Conv3d(inter_channel, out_channel, 1, bias=False))

    def forward(self, x):
        size = x.size()[2:]
        x0 = self.trans_layer(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        x1 = self.conv1_1(x0)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), size, mode='trilinear', align_corners=True)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), size, mode='trilinear', align_corners=True)
        out1 = self.conv2_6(F.relu(x2_1 + x2_2 + x2_3, inplace=True))
        if self.is_dynamic_empty_cache:
            del x1, x2_1, x2_2, x2_3
            torch.cuda.empty_cache()

        x2 = self.conv1_2(x0)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), size, mode='trilinear', align_corners=True)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), size, mode='trilinear', align_corners=True)
        x2_6 = F.interpolate(self.conv2_5(self.pool5(x2)), size, mode='trilinear', align_corners=True)
        out2 = self.conv2_7(F.relu(x2_4 + x2_5 + x2_6, inplace=True))
        if self.is_dynamic_empty_cache:
            del x2, x2_4, x2_5, x2_6
            torch.cuda.empty_cache()

        out = self.conv3(torch.cat([out1, out2], dim=1))
        out = F.relu(x0 + out, inplace=True)
        if self.is_dynamic_empty_cache:
            del x0, out1, out2
            torch.cuda.empty_cache()

        out = self.score_layer(out)

        return out

class ResBaseConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, p=0.2, stride=1, is_identify=True, is_dynamic_empty_cache=False):
        """residual base block, including two layer convolution, instance normalization, drop out and leaky ReLU"""
        super(ResBaseConvBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.residual_unit = nn.Sequential(
            _ConvINReLU3D(in_channel, out_channel, 3, stride=stride, padding=1, p=p),
            _ConvIN3D(out_channel, out_channel, 3, stride=1, padding=1))
        self.shortcut_unit = nn.Sequential() if stride == 1 and in_channel == out_channel and is_identify else \
            _ConvIN3D(in_channel, out_channel, 1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output = self.relu(output)

        return output

class AnisotropicConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, p=0.2, stride=1, is_identify=True, is_dynamic_empty_cache=False):
        """Anisotropic convolution block, including two layer convolution,
         instance normalization, drop out and ReLU"""
        super(AnisotropicConvBlock, self).__init__()
        self.is_dynamic_empty_cache = is_dynamic_empty_cache
        self.residual_unit = nn.Sequential(
            _ConvINReLU3D(in_channel, out_channel, kernel_size=(3, 3, 1), stride=stride, padding=(1, 1, 0), p=p),
            _ConvIN3D(out_channel, out_channel, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1)))
        self.shortcut_unit = nn.Sequential() if stride == 1 and in_channel == out_channel and is_identify else \
            _ConvIN3D(in_channel, out_channel, kernel_size=1, stride=stride, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.residual_unit(x)
        output += self.shortcut_unit(x)
        if self.is_dynamic_empty_cache:
            del x
            torch.cuda.empty_cache()

        output = self.relu(output)

        return output

class EfficientSegNet(nn.Module):

    def __init__(self, in_ch, channels=16, patch_size=(192,192,192)):
        super().__init__()

        # EfficientSegNet parameter.
        num_channel = [channels, channels*2, channels*4, channels*8, channels*16]
        num_blocks = [2, 2, 2, 2]
        decoder_num_block = 1
        self.num_depth = 5
        self.auxiliary_task = False
        self.auxiliary_class = 1
        self.is_dynamic_empty_cache = False


        encoder_conv_block = ResBaseConvBlock
        decoder_conv_block = AnisotropicConvBlock
        context_block = AnisotropicAvgPooling
        # context_block = None


        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = self._mask_layer(encoder_conv_block, in_ch, num_channel[0], num_blocks[0], stride=1)
        self.conv1_0 = self._mask_layer(encoder_conv_block, num_channel[0], num_channel[1], num_blocks[0], stride=2)
        self.conv2_0 = self._mask_layer(encoder_conv_block, num_channel[1], num_channel[2], num_blocks[1], stride=2)
        self.conv3_0 = self._mask_layer(encoder_conv_block, num_channel[2], num_channel[3], num_blocks[2], stride=2)
        self.conv4_0 = self._mask_layer(encoder_conv_block, num_channel[3], num_channel[4], num_blocks[3], stride=2)

        if context_block is not None:
            context_kernel_size = [i // 16 for i in patch_size]
            self.context_block = context_block(num_channel[4], num_channel[4], kernel_size=context_kernel_size,
                                               is_dynamic_empty_cache=self.is_dynamic_empty_cache)
        else:
            self.context_block = nn.Sequential()

        self.trans_4 = _ConvINReLU3D(num_channel[4], num_channel[3], kernel_size=1, stride=1, padding=0, p=0.2)
        self.trans_3 = _ConvINReLU3D(num_channel[3], num_channel[2], kernel_size=1, stride=1, padding=0, p=0.2)
        self.trans_2 = _ConvINReLU3D(num_channel[2], num_channel[1], kernel_size=1, stride=1, padding=0, p=0.2)
        self.trans_1 = _ConvINReLU3D(num_channel[1], num_channel[0], kernel_size=1, stride=1, padding=0, p=0.2)

        self.conv3_1 = self._mask_layer(decoder_conv_block, num_channel[3],
                                        num_channel[3], decoder_num_block, stride=1)
        self.conv2_2 = self._mask_layer(decoder_conv_block, num_channel[2],
                                        num_channel[2], decoder_num_block, stride=1)
        self.conv1_3 = self._mask_layer(decoder_conv_block, num_channel[1],
                                        num_channel[1], decoder_num_block, stride=1)
        self.conv0_4 = self._mask_layer(decoder_conv_block, num_channel[0],
                                        num_channel[0], decoder_num_block, stride=1)

        self._initialize_weights()
        # self.final.bias.data.fill_(-2.19)

    def _mask_layer(self, block, in_channels, out_channels, num_block, stride):
        layers = []
        layers.append(block(in_channels, out_channels, p=0.2, stride=stride, is_identify=False,
                            is_dynamic_empty_cache=self.is_dynamic_empty_cache))
        for _ in range(num_block-1):
            layers.append(block(out_channels, out_channels, p=0.2, stride=1, is_identify=True,
                                is_dynamic_empty_cache=self.is_dynamic_empty_cache))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0_0(x)
        x1_0 = self.conv1_0(x)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        x4_0 = self.context_block(x4_0)

        x3_0 = self.conv3_1(self.up(self.trans_4(x4_0)) + x3_0)
        if self.is_dynamic_empty_cache:
            del x4_0
            torch.cuda.empty_cache()
        x2_0 = self.conv2_2(self.up(self.trans_3(x3_0)) + x2_0)
        if self.is_dynamic_empty_cache:
            del x3_0
            torch.cuda.empty_cache()

        x1_0 = self.conv1_3(self.up(self.trans_2(x2_0)) + x1_0)
        if self.is_dynamic_empty_cache:
            del x2_0
            torch.cuda.empty_cache()

        x = self.conv0_4(self.up(self.trans_1(x1_0)) + x)
        if self.is_dynamic_empty_cache:
            del x1_0
            torch.cuda.empty_cache()
        
        
        return x


