import torch
import torch.nn as nn
import torch.nn.functional as F

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
            astrous_rates = [0, 2, 4, 8]
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
        output = self.bottleneck(output)
        return output

class UnetConv3(nn.Module):
    def __init__(self, inplanes, outplanes, is_batchnorm, kernel_size=3, stride=1, padding_size=1):
        super(UnetConv3, self).__init__()
        
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(inplanes, outplanes, kernel_size, stride=stride, padding=padding_size), 
                                       nn.BatchNorm3d(outplanes),
                                       nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv3d(outplanes, outplanes, kernel_size, stride=1, padding=padding_size), 
                                       nn.BatchNorm3d(outplanes),
                                       nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(inplanes, outplanes, kernel_size, stride=stride, padding=padding_size), 
                                       nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(nn.Conv3d(outplanes, outplanes, kernel_size, stride=1, padding=padding_size), 
                                       nn.ReLU(inplace=True))
    
    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        return out

class UnetUp3D(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3D, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=3, padding_size=1)
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

    def forward(self, input1, input2):
        outputs2 = self.up(input2)
        offset = outputs2.size()[2] - input1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(input1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

class UnetDsv3D(nn.Module):
    def __init__(self, inplanes, outplanes, scale_factor):
        super(UnetDsv3D, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(inplanes, outplanes, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'),)
    def forward(self, input):
        return self.dsv(input)

class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')


    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock2D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=2, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )


class GridAttentionBlock3D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock3D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=3, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )

class SingleAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(SingleAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)

        return self.combine_gates(gate_1), attention_1

class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)

class AttentionUnet3D(nn.Module):
    def __init__(self, in_ch, channels=16, nonlocal_mode='concatenation', attention_block='single', 
                 attention_downsample=2, is_batchnorm=True, is_dsv=False):
        super(AttentionUnet3D, self).__init__()
        self.in_channels = in_ch
        self.is_batchnorm = is_batchnorm
        self.is_dsv = is_dsv

        self.conv1 = UnetConv3(self.in_channels, channels, self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = UnetConv3(channels, channels * 2, self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = UnetConv3(channels * 2, channels * 4, self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = UnetConv3(channels * 4, channels * 8, self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        self.center = UnetConv3(channels * 8, channels * 16, self.is_batchnorm)
        if self.is_batchnorm:
            self.gate = nn.Sequential(nn.Conv3d(channels * 16, channels * 16, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm3d(channels * 16),
                                      nn.ReLU(inplace=True))
        else:
            self.gate = nn.Sequential(nn.Conv3d(channels * 16, channels * 16, kernel_size=1, stride=1, padding=0),
                                      nn.ReLU(inplace=True))
        if attention_block == 'single':
            self.attention_block = SingleAttentionBlock
        elif attention_block == 'multi':
            self.attention_block = MultiAttentionBlock

        self.attentionblock2 = self.attention_block(in_size=channels * 2, gate_size=channels * 4, inter_size=channels * 2, 
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_downsample)
        self.attentionblock3 = self.attention_block(in_size=channels * 4, gate_size=channels * 8, inter_size=channels * 4, 
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_downsample)
        self.attentionblock4 = self.attention_block(in_size=channels * 8, gate_size=channels * 16, inter_size=channels * 8, 
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_downsample)

        self.up_concat1 = UnetUp3D(channels * 2, channels * 1, self.is_batchnorm)
        self.up_concat2 = UnetUp3D(channels * 4, channels * 2, self.is_batchnorm)
        self.up_concat3 = UnetUp3D(channels * 8, channels * 4, self.is_batchnorm)
        self.up_concat4 = UnetUp3D(channels * 16, channels * 8, self.is_batchnorm)

        self.dsv2 = UnetDsv3D(channels * 2, channels, 2)
        self.dsv3 = UnetDsv3D(channels * 4, channels, 4)
        self.dsv4 = UnetDsv3D(channels * 8, channels, 8)
        self.final = nn.Conv3d(channels * 4, channels, 1, stride=1, padding=0)

        self.aspp = ASPP(channels * 16, channels * 16)
        
    def forward(self, input):
        c1 = self.conv1(input)
        maxpool1 = self.maxpool1(c1)

        c2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(c2)

        c3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(c3)

        c4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(c4)

        c5 = self.center(maxpool4)
        # gate = self.gate(c5)
        gate = self.aspp(c5)

        g_c4, _ = self.attentionblock4(c4, gate)
        up4 = self.up_concat4(g_c4, c5)

        g_c3, _ = self.attentionblock3(c3, up4)
        up3 = self.up_concat3(g_c3, up4)

        g_c2, _ = self.attentionblock2(c2, up3)
        up2 = self.up_concat2(g_c2, up3)
        
        up1 = self.up_concat1(c1, up2)
        
        # if self.is_dsv:
        #     dsv1 = up1
        #     dsv2 = self.dsv2(up2)
        #     dsv3 = self.dsv3(up3)
        #     dsv4 = self.dsv4(up4)
        #     out = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
        # else:
        out = up1
        return out