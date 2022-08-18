import os
import sys
import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from scipy.ndimage.interpolation import zoom

class SegConfig:

    def __init__(self,  network_f):
        # TODO: 模型配置文件
        self.network_f = network_f
        if self.network_f is not None:
            from mmcv import Config

            if isinstance(self.network_f, str):
                self.network_cfg = Config.fromfile(self.network_f)
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as temp_config_dir:
                    with tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix='.py') as temp_config_file:
                        with open(temp_config_file.name, 'wb') as f:
                            f.write(self.network_f.read())

                        self.network_cfg = Config.fromfile(temp_config_file.name)

    def __repr__(self) -> str:
        return str(self.__dict__)


class SegModel:

    def __init__(self, model_f, network_f):
        # TODO: 模型文件定制
        self.model_f = model_f
        self.network_f = network_f


class SegPredictor:

    def __init__(self, gpu: int, model: SegModel):
        self.gpu = gpu
        self.model = model
        self.config = SegConfig(self.model.network_f)
        self.load_model()

    def load_model(self):
        self.net = self._load_model(self.model.model_f, self.config.network_cfg, half=False)

    def _load_model(self, model_f, network_f, half=False) -> None:
        if isinstance(model_f, str):
            net = self.load_model_pth(model_f, network_f, half)
        else:
            model_f.seek(0)
            headers = model_f.peek(2)
            net = self.load_model_pth(model_f, network_f, half)
        return net
    
    def load_model_pth(self, model_f, network_cfg, half) -> None:
        # 加载动态图
        config = network_cfg
        if config.model["backbone"] == "ResUnet":
            backbone = ResUnet(config.model["in_ch"], channels=config.model["channels"])
        elif config.model["backbone"] == "EfficientSegNet":
            from network.effnet import EfficientSegNet
            backbone = EfficientSegNet(
                config.in_ch, 
                channels=config.model["channels"],
                patch_size=config.patch_size,
            )
        elif config.model["backbone"] == "AttentionUnet3D":
            from network.AttentionUnet import AttentionUnet3D
            backbone = AttentionUnet3D(
                config.in_ch, 
                channels=config.model["channels"],
                attention_block='multi',
            )
        else:
            raise TypeError("<<<<<<<<<<<wrong backbone>>>>>>>>>>>>")

        if config.model["head_type"] == "soft":
            head = SegSoftHead(in_channels=config.model["channels"], classes=config.model["classes"])
        elif config.model["head_type"] == "sig":
            head = SegSigHead(in_channels=config.model["channels"], classes=1)
        else:
            raise TypeError("<<<<<<<<<<<wrong head>>>>>>>>>>>>")
            
        net = SegNetwork(backbone, head)
        checkpoint = torch.load(model_f, map_location=f"cpu")
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        net.eval()
        net.half()
        net.cuda()
        net = net.forward_test
        return net

    def _get_input(self, vol, spacing_zyx):

        config = self.config.network_cfg

        def _window_array(vol, win_level, win_width):
            win = [
                win_level - win_width / 2,
                win_level + win_width / 2,
            ]
            vol = torch.clamp(vol, win[0], win[1])
            vol -= win[0]
            vol /= win_width
            return vol

        vol = torch.from_numpy(vol).float()[None, None]
        vol = [_window_array(vol, wl, wd) for wl, wd in zip(config.win_level, config.win_width)]
        vol = torch.cat(vol, dim=1)
        vol_shape = np.array(vol.shape[2:], dtype=np.float32)
        patch_size = np.array(config.patch_size)
        if np.any(vol_shape != patch_size):
            vol = torch.nn.functional.interpolate(
                vol.float(), size=tuple(patch_size), mode="trilinear", align_corners=False
            )
        vol = vol.detach()
        return vol

    def forward(self, vol, spacing, sup_mask=None):

        config = self.config.network_cfg
        patch_size = np.array(config.patch_size)
        ori_shape = np.array(vol.shape)
        spacing_zyx = np.array(spacing)
        data = self._get_input(vol, spacing_zyx)
        
        with autocast():
            data = data.cuda().detach()
            pred_seg = self.net(data)
            del data
            if pred_seg.size()[1] > 1:
                pred_seg = F.softmax(pred_seg, dim=1)
                pred_seg = torch.argmax(pred_seg, dim=1, keepdim=True)
            else:
                pred_seg = torch.sigmoid(pred_seg)
                pred_seg[pred_seg >= config.threshold] = 1
                pred_seg[pred_seg < config.threshold] = 0
     
        heatmap = pred_seg.cpu().detach().numpy()[0, 0].astype(np.int8)
        heatmap = zoom(heatmap, np.array(ori_shape / np.array(patch_size)), order=0)

        heatmap = heatmap.astype(np.uint8)

        return heatmap

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        del identity
        del x
        torch.cuda.empty_cache()
        
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
    def __init__(self, in_channels, out_channels, output_stride=16):
        super(ASPP, self).__init__()

        if output_stride == 16:
            astrous_rates = [0, 4, 8, 12]
        elif output_stride == 8:
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

        del input
        del input1
        del input2
        del input3
        del input4
        del input5
        torch.cuda.empty_cache()

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



class SegSigHead(nn.Module):

    def __init__(self, in_channels, classes=1):
        super(SegSigHead, self).__init__()
        self.conv = nn.Conv3d(in_channels, 1, 1)
        self.bce_loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, inputs):
        features = self.conv(inputs)
        return features

    def forward_test(self, inputs):
        return self.forward(inputs)

class SegSoftHead(nn.Module):

    def __init__(self, in_channels, classes=14):
        super(SegSoftHead, self).__init__()
        self.conv = nn.Conv3d(in_channels, classes, 1)
        self.multi_loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        self._classes = classes

    def forward(self, inputs):
        seg_predict = self.conv(inputs)
        return seg_predict

    def forward_test(self, inputs):
        return self.forward(inputs)

class SegNetwork(nn.Module):

    def __init__(self, backbone,
                head, apply_sync_batchnorm=False,
                train_cfg=None,
                test_cfg=None
                ):
        super(SegNetwork, self).__init__()
        self.backbone = backbone
        self.head = head
        self._show_count = 0
        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.ignore
    def forward(self, vol, seg):
        vol = vol.float()
        seg = seg.float()
        features = self.backbone(vol)
        head_outs = self.head(features)
        del features
        del vol
        del seg
        torch.cuda.empty_cache()
        loss = self.head.loss(head_outs, seg)
        return loss

    @torch.jit.export
    def forward_test(self, img):
        features = self.backbone(img)
        del img 
        torch.cuda.empty_cache()
        seg_predict = self.head.forward_test(features)
        del features
        torch.cuda.empty_cache()
        return seg_predict

    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)
