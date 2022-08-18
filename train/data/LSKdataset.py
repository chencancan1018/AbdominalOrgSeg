import os
import monai
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from .transforms import MonaiElasticDictTransform

class LSKSegDataset(Dataset):
    def __init__(
        self,
        config,
    ):
        self._patch_size = config["patch_size"]
        self._data_file_list = self._load_file_list(
            config["data_root"], 
            config["dst_list_file"], 
            config["sample_frequent"],
        )
        self._win_level = config["win_level"]
        self._win_width = config["win_width"]
        self.aug_paras = config["aug_paras"]

    def _load_file_list(self, data_root, dst_list_file, sample_frequent):
        data_file_list = []
        with open(dst_list_file, "r") as f:
            for line in f:
                line = line.strip().split(' ')
                file_name = line[0]
                file_name = os.path.join(data_root, file_name)
                if not os.path.exists(file_name):
                    print(f"{line} not exist")
                    continue
                for _ in range(sample_frequent):
                    data_file_list.append(file_name)
        random.shuffle(data_file_list)
        assert len(data_file_list) != 0, "has no avilable file in dst_list_file"
        return data_file_list
        
    def find_valid_region(self, mask, values, low_margin=[0,0,0], up_margin=[0,0,0]):
        for v in values:
            mask[mask == v] = 100
        nonzero_points = np.argwhere((mask > 20))
        if len(nonzero_points) == 0:
            return None, None
        else:
            v_min = np.min(nonzero_points, axis=0)
            v_max = np.max(nonzero_points, axis=0)
            assert len(v_min) == len(low_margin), f'the length of margin is not equal the mask dims {len(v_min)}!'
            for idx in range(len(v_min)):
                v_min[idx] = max(0, v_min[idx] - low_margin[idx])
                v_max[idx] = min(mask.shape[idx], v_max[idx] + up_margin[idx])
            return v_min, v_max

    def window_array(self, vol, win_level, win_width):
        win = [
            win_level - win_width / 2,
            win_level + win_width / 2,
        ]
        vol = torch.clamp(vol, win[0], win[1])
        vol -= win[0]
        vol /= win_width
        return vol

    def __getitem__(self, index):
        
        data = np.load(self._data_file_list[index])
        result = {}
        with torch.no_grad():
            vol = data["vol"]
            seg = data["seg"]

            # for liver, kidney and spleen
            rand_z = random.randint(4, 7); rand_yx = random.randint(20, 50)
            pmin, pmax = self.find_valid_region(seg.copy(), values=[1,2,3,13], low_margin=[rand_z,rand_yx,rand_yx], up_margin=[rand_z,rand_yx,rand_yx])
            vol = vol[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]]
            seg = seg[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]]
            seg_ = np.zeros(seg.shape, dtype=np.uint8)
            seg_[seg == 1] = 1; seg_[seg == 2] = 2; seg_[seg == 3] = 3; seg_[seg == 13] = 4

            vol = torch.from_numpy(vol).float()[None, None]
            vol_shape = np.array(vol.shape[2:], dtype=np.float32)
            seg = torch.from_numpy(seg_).int()[None, None]

            # 加窗
            vol = [self.window_array(vol, wl, wd) for wl, wd in zip(self._win_level, self._win_width)]
            vol = torch.cat(vol, dim=1)

            # resize
            patch_size = np.array(self._patch_size)
            if np.any(vol_shape != patch_size):
                vol = torch.nn.functional.interpolate(
                    vol.float(), size=self._patch_size, mode="trilinear", align_corners=False, 
                )
            vol = vol[0] # channel first 
            seg_shape = np.array(seg.shape[2:], dtype=np.float32)
            if np.any(seg_shape != patch_size):
                seg = torch.nn.functional.interpolate(
                    seg.float(), size=self._patch_size, mode="nearest"
                )[0, 0]
            seg = seg[None] # channel first

            # augmentation
            if self.aug_paras:
                aug = MonaiElasticDictTransform(self.aug_paras)
                vol, seg = aug(data=(vol, seg))

            result['vol'] = vol.detach()
            result['seg'] = seg.detach()
        del data
        return result

    def __len__(self):
        return len(self._data_file_list)
