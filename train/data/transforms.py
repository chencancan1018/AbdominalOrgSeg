import torch
import torchvision
import numpy as np

from monai.transforms.croppad.array import RandScaleCrop
from monai.transforms.croppad.dictionary import RandScaleCropD
from monai.transforms.intensity.array import (
    RandGaussianSharpen, 
    RandShiftIntensity,
    RandScaleIntensity,
    ScaleIntensityRange,
)
from monai.transforms.spatial.array import (
    RandRotate90, 
    RandFlip, 
    RandAffine, 
    Rand3DElastic
)
from monai.transforms.spatial.dictionary import Rand3DElasticD
from monai.transforms.utility.array import ToTensor
from monai.transforms.utility.dictionary import ToTensorD
from monai.transforms.spatial.dictionary import (
    RandRotate90d,
    RandFlipd,
)

class MonaiElasticDictTransform():
    
    def __init__(self, aug_parameters: dict):

        self.patch_size = aug_parameters.setdefault("patch_size", (128, 128, 128))
        self.roi_scale = aug_parameters.setdefault("roi_scale", (1.0, 1.0, 1.0)) 
        self.max_roi_scale = aug_parameters.setdefault('max_roi_scale', (1.0, 1.0, 1.0)) 
        self.rotate_x = aug_parameters.setdefault('rot_range_x', (-np.pi/9, np.pi/9))
        self.rotate_y = aug_parameters.setdefault('rot_range_y', (-np.pi/9, np.pi/9)) 
        self.rotate_z = aug_parameters.setdefault('rot_range_z', (-np.pi/9, np.pi/9)) 
        self.rotate_90 = aug_parameters.setdefault('rot_90', False)
        self.flip = aug_parameters.setdefault("flip", False)
        self.prob = aug_parameters.setdefault('prob', 0.5) 
        self.bright_bias = aug_parameters.setdefault('bright_bias', (-0.2, 0.2)) 
        self.bright_w = aug_parameters.setdefault('bright_weight', (-0.2, 0.2)) 
        self.translate_x = aug_parameters.setdefault('translate_x', (-5.0, 5.0))
        self.translate_y = aug_parameters.setdefault('translate_y', (-5.0, 5.0))
        self.translate_z = aug_parameters.setdefault('translate_z', (-5.0, 5.0))
        self.scale_x = aug_parameters.setdefault('scale_x', (-0.1, 0.1)) 
        self.scale_y = aug_parameters.setdefault('scale_y', (-0.1, 0.1)) 
        self.scale_z = aug_parameters.setdefault('scale_z', (-0.1, 0.1)) 
        self.elastic_sigma_range =  aug_parameters.setdefault('elastic_sigma_range', (3, 5))
        self.elastic_magnitude_range = aug_parameters.setdefault('elastic_magnitude_range', (100, 200))

        aug_dict = list()
        aug_img = list()

        # random crop
        if self.roi_scale != self.max_roi_scale:
            rand_crop = RandScaleCropD(keys=["image", "label"], roi_scale=self.roi_scale, max_roi_scale=self.max_roi_scale)
            aug_dict.append(rand_crop)
        
        if self.rotate_90:
            rand_rotate90d = RandRotate90d(keys=["image", "label"], prob=self.prob, max_k=3)
            aug_dict.append(rand_rotate90d)
        
        if self.flip:
            rand_flipd_x = RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=self.prob) 
            rand_flipd_y = RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=self.prob)
            rand_flipd_z = RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=self.prob)
            aug_dict.append(rand_flipd_x)
            aug_dict.append(rand_flipd_y)
            aug_dict.append(rand_flipd_z)

        rand_elastic = Rand3DElasticD(
            keys=["image", "label"],
            sigma_range=self.elastic_sigma_range,
            magnitude_range=self.elastic_magnitude_range,
            prob=self.prob,
            rotate_range=(self.rotate_x, self.rotate_y, self.rotate_z),
            shear_range=None,
            translate_range=(self.translate_x, self.translate_y, self.translate_z),
            scale_range=(self.scale_x, self.scale_y, self.scale_z),
            spatial_size=self.patch_size,
            mode=("bilinear", "nearest"), #'nearest' for label,
            padding_mode='border',
        )
        aug_dict.append(rand_elastic)

        gaussian_blur = RandGaussianSharpen()
        aug_img.append(gaussian_blur)
 
        intensity_shift = RandShiftIntensity(offsets=self.bright_bias, prob=self.prob)
        intensity_scale = RandScaleIntensity(factors=self.bright_w, prob=self.prob)
        aug_img.append(intensity_shift)
        aug_img.append(intensity_scale)

        # clip
        clip_0_1 = ScaleIntensityRange(a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True)
        aug_img.append(clip_0_1)
        
        array_to_tensor = ToTensor(dtype=torch.float32)
        dict_to_tensor = ToTensorD(keys=["image", "label"], dtype=torch.float32)
        aug_img.append(array_to_tensor)
        aug_dict.append(dict_to_tensor)

        self.aug_dict = torchvision.transforms.Compose(aug_dict)
        self.aug_img = torchvision.transforms.Compose(aug_img)
    
    def __call__(self, data):
        img, mask = data
        data_dict = {'image': img, 'label': mask}
        data_dict = self.aug_dict(data_dict)

        img = data_dict["image"]
        mask = data_dict['label']
        img = self.aug_img(img)
        data = img, mask
        return data