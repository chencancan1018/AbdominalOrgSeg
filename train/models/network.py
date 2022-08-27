import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegSigHead(nn.Module):

    def __init__(self, in_channels, patch_size=[128, 128, 128], is_aux=False):
        super(SegSigHead, self).__init__()
        self.conv = nn.Conv3d(in_channels, 1, 1)
        self.conv1 = nn.Conv3d(in_channels, 1, 1)
        self.conv2 = nn.Conv3d(2 * in_channels, 1, 1)
        self.conv3 = nn.Conv3d(4 * in_channels, 1, 1)
        self.bce_loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)
        self.is_aux = is_aux
        self._patch_size = patch_size

    def forward(self, inputs):
        if self.is_aux:
            features = self.conv(inputs[0])
            features1 = self.conv1(inputs[1])
            features2 = self.conv2(inputs[2])
            features3 = self.conv3(inputs[3])
            seg_predict = [features, features1, features2, features3]
            return seg_predict
        else:
            seg_predict = self.conv(inputs)
            return seg_predict

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

            # patch_size = [128, 128, 128]  
            shape = tuple([int(v // 2)  for v in self._patch_size])
            seg1 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").float()
            target1 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").float()
            seg_tp1 = (seg1 >= 0.5) 
            seg_tn1 = (seg1 < 0.5) * 1 
            seg_tp_sum1 = ((seg1 >= 0.5).sum() + 1)
            seg_tn_sum1 = ((seg1 < 0.5).sum() + 1)

            shape = tuple([int(v // 4)  for v in self._patch_size])
            seg2 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").float()
            target2 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").float()
            seg_tp2 = (seg2 >= 0.5) 
            seg_tn2 = (seg2 < 0.5) * 1 
            seg_tp_sum2 = ((seg2 >= 0.5).sum() + 1)
            seg_tn_sum2 = ((seg2 < 0.5).sum() + 1)

            shape = tuple([int(v // 8)  for v in self._patch_size])
            seg3 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").float()
            target3 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").float()
            seg_tp3 = (seg3 >= 0.5) 
            seg_tn3 = (seg3 < 0.5) * 1 
            seg_tp_sum3 = ((seg3 >= 0.5).sum() + 1)
            seg_tn_sum3 = ((seg3 < 0.5).sum() + 1)

        if self.is_aux:
            loss = self.bce_loss_func(seg_predict[0], seg)
            dice_loss = self._dice_loss(seg_predict[0], target)
        else:
            loss = self.bce_loss_func(seg_predict, seg)
            dice_loss = self._dice_loss(seg_predict, target)
        loss_pos = (loss * seg_tp).sum() / seg_tp_sum
        loss_neg = (loss * seg_tn).sum() / seg_tn_sum
        pos_ = loss_pos
        neg_ = loss_neg
        dice_ = dice_loss

        if self.is_aux:
            loss = self.bce_loss_func(seg_predict[1], seg1)
            loss_pos = (loss * seg_tp1).sum() / seg_tp_sum1
            loss_neg = (loss * seg_tn1).sum() / seg_tn_sum1
            dice_loss = self._dice_loss(seg_predict[1], target1)
            pos_ += (1/2) * loss_pos
            neg_ += (1/2) * loss_neg
            dice_ += (1/2) * dice_loss

            loss = self.bce_loss_func(seg_predict[2], seg2)
            loss_pos = (loss * seg_tp2).sum() / seg_tp_sum2
            loss_neg = (loss * seg_tn2).sum() / seg_tn_sum2
            dice_loss = self._dice_loss(seg_predict[2], target2)
            pos_ += (1/4) * loss_pos
            neg_ += (1/4) * loss_neg
            dice_ += (1/4) * dice_loss

            loss = self.bce_loss_func(seg_predict[3], seg3)
            loss_pos = (loss * seg_tp3).sum() / seg_tp_sum3
            loss_neg = (loss * seg_tn3).sum() / seg_tn_sum3
            dice_loss = self._dice_loss(seg_predict[3], target3)
            pos_ += (1/8) * loss_pos
            neg_ += (1/8) * loss_neg
            dice_ += (1/8) * dice_loss
        return {'loss_pos': pos_, 'loss_neg': neg_, 'dice_loss': dice_}

class SegSoftHead(nn.Module):

    def __init__(self, in_channels, classes=14, patch_size=[128, 128, 128], is_aux=False):
        super(SegSoftHead, self).__init__()
        self.conv = nn.Conv3d(in_channels, classes, 1)
        self.conv1 = nn.Conv3d(in_channels, classes, 1)
        self.conv2 = nn.Conv3d(2 * in_channels, classes, 1)
        self.conv3 = nn.Conv3d(4 * in_channels, classes, 1)
        self.multi_loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        self.is_aux = is_aux
        self._patch_size = patch_size
        self._classes = classes

    def forward(self, inputs):
        if self.is_aux:
            features = self.conv(inputs[0])
            features1 = self.conv1(inputs[1])
            features2 = self.conv2(inputs[2])
            features3 = self.conv3(inputs[3])
            seg_predict = [features, features1, features2, features3]
            return seg_predict
        else:
            seg_predict = self.conv(inputs)
            return seg_predict

    def _dice_loss(self, logits, target):
        eps = 1e-9
        pred = F.softmax(logits, dim=1)
        target = F.one_hot(target.long(), num_classes=self._classes).squeeze().permute(0, 4, 1, 2, 3)
        tp = (pred * target).sum((0, 2, 3, 4))
        fp = (pred * (1 - target)).sum((0, 2, 3, 4))
        fn = ((1 - pred) * target).sum((0, 2, 3, 4))
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        dice = dice.mean()
        return 1 - dice

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
            
            if self.is_aux:
                # patch_size = [128, 128, 128]  
                shape = tuple([int(v // 2)  for v in self._patch_size])
                seg1 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")[:, 0, ::].long()
                target1 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")
                seg_tp1 = (seg1 >= 0.5)
                seg_tn1 = (seg1 < 0.5) * 1 
                seg_tp_sum1 = ((seg1 >= 0.5).sum() + 1)
                seg_tn_sum1 = ((seg1 < 0.5).sum() + 1)

                shape = tuple([int(v // 4)  for v in self._patch_size])
                seg2 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")[:, 0, ::].long()
                target2 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")
                seg_tp2 = (seg2 >= 0.5) 
                seg_tn2 = (seg2 < 0.5) * 1 
                seg_tp_sum2 = ((seg2 >= 0.5).sum() + 1)
                seg_tn_sum2 = ((seg2 < 0.5).sum() + 1)

                shape = tuple([int(v // 8)  for v in self._patch_size])
                seg3 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")[:, 0, ::].long()
                target3 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")
                seg_tp3 = (seg3 >= 0.5) 
                seg_tn3 = (seg3 < 0.5) * 1 
                seg_tp_sum3 = ((seg3 >= 0.5).sum() + 1)
                seg_tn_sum3 = ((seg3 < 0.5).sum() + 1)

        if self.is_aux:
            loss = self.multi_loss_func(seg_predict[0], seg)
            dice_loss = self._dice_loss(seg_predict[0], target)
        else:
            loss = self.multi_loss_func(seg_predict, seg)
            dice_loss = self._dice_loss(seg_predict, target)
        loss_pos = (loss * seg_tp).sum() / seg_tp_sum
        loss_neg = (loss * seg_tn).sum() / seg_tn_sum
        pos_ = loss_pos
        neg_ = loss_neg
        dice_ = dice_loss

        if self.is_aux:
            loss = self.multi_loss_func(seg_predict[1], seg1)
            loss_pos = (loss * seg_tp1).sum() / seg_tp_sum1
            loss_neg = (loss * seg_tn1).sum() / seg_tn_sum1
            dice_loss = self._dice_loss(seg_predict[1], target1)
            pos_ += (1/2) * loss_pos
            neg_ += (1/2) * loss_neg
            dice_ += (1/2) * dice_loss

            loss = self.multi_loss_func(seg_predict[2], seg2)
            loss_pos = (loss * seg_tp2).sum() / seg_tp_sum2
            loss_neg = (loss * seg_tn2).sum() / seg_tn_sum2
            dice_loss = self._dice_loss(seg_predict[2], target2)
            pos_ += (1/4) * loss_pos
            neg_ += (1/4) * loss_neg
            dice_ += (1/4) * dice_loss

            loss = self.multi_loss_func(seg_predict[3], seg3)
            loss_pos = (loss * seg_tp3).sum() / seg_tp_sum3
            loss_neg = (loss * seg_tn3).sum() / seg_tn_sum3
            dice_loss = self._dice_loss(seg_predict[3], target3)
            pos_ += (1/8) * loss_pos
            neg_ += (1/8) * loss_neg
            dice_ += (1/8) * dice_loss

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