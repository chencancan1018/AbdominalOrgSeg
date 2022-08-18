import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegSigHead(nn.Module):

    def __init__(self, in_channels, classes=1):
        super(SegSigHead, self).__init__()
        self.conv = nn.Conv3d(in_channels, 1, 1)
        self.bce_loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, inputs):
        features = self.conv(inputs)
        return features

    def _dice_loss(self, logits, target):
        eps = 1e-9
        pred = torch.sigmoid(logits)
        inter_section = (pred * target).sum()
        inter_section = 2 * inter_section + eps
        union = pred.sum() + target.sum() + eps
        dice = inter_section / union
        return 1 - dice
    
    def loss(self, inputs, target):
        seg_predict = inputs
        with torch.no_grad():
            seg = target 
            seg = seg.float()

            seg_tp = (seg >= 0.5) 
            seg_tn = (seg < 0.5) * 1 
            seg_tp_sum = ((seg >= 0.5).sum() + 1)
            seg_tn_sum = ((seg < 0.5).sum() + 1)

        loss = self.bce_loss_func(seg_predict, seg)
        loss_pos = (loss * seg_tp).sum() / seg_tp_sum
        loss_neg = (loss * seg_tn).sum() / seg_tn_sum
        dice_loss = self._dice_loss(seg_predict, target)
        pos_ = loss_pos
        neg_ = loss_neg
        dice_ = dice_loss
        return {'loss_pos': pos_, 'loss_neg': neg_, 'dice_loss': dice_}

class SegSoftHead(nn.Module):

    def __init__(self, in_channels, classes=14):
        super(SegSoftHead, self).__init__()
        self.conv = nn.Conv3d(in_channels, classes, 1)
        self.multi_loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        self._classes = classes

    def forward(self, inputs):
        seg_predict = self.conv(inputs)
        return seg_predict

    def _dice_loss(self, logits, target):
        eps = 1e-9
        pred = F.softmax(logits, dim=1)
        inter_section = torch.sum(pred * target, dim=(2,3,4))
        inter_section = 2 * inter_section 
        union = torch.sum(pred, dim=(2,3,4)) + torch.sum(target, dim=(2,3,4))
        dice = (inter_section + eps) / (union + eps)
        dice_loss = 1 - dice.mean(1)
        dice_loss = dice_loss.mean()
        return dice_loss

    def loss(self, inputs, target):
        seg_predict = inputs
        with torch.no_grad():
            seg = target 
            seg = seg[:, 0, ::].long()
            target = (target > 0.5) * 1
        
            seg_tp = (seg >= 0.5) 
            seg_tn = (seg < 0.5) * 1
            seg_tp_sum = ((seg >= 0.5).sum() + 1)
            seg_tn_sum = ((seg < 0.5).sum() + 1)

        loss = self.multi_loss_func(seg_predict, seg)
        loss_pos = (loss * seg_tp).sum() / seg_tp_sum
        loss_neg = (loss * seg_tn).sum() / seg_tn_sum
        dice_loss = self._dice_loss(seg_predict, target)
        pos_ = loss_pos
        neg_ = loss_neg
        dice_ = dice_loss
        return {'loss_pos': pos_, 'loss_neg': neg_, 'dice_loss': dice_}

class SegNetwork(nn.Module):

    def __init__(self, backbone, head, 
                apply_sync_batchnorm=False,
                ):
        super(SegNetwork, self).__init__()
        self.backbone = backbone
        self.head = head
        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.ignore
    def forward(self, vol, seg):
        vol = vol.float()
        seg = seg.float()
        features = self.backbone(vol)
        head_outs = self.head(features)
        loss = self.head.loss(head_outs, seg)
        return loss

    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)