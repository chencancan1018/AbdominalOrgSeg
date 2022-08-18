"""生成模型输入数据."""

import argparse
import glob
import json
import os
import random
import sys
import traceback
from cv2 import findCirclesGrid

import numpy as np
import SimpleITK as sitk
import threadpool
import threading
from queue import Queue
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser()
  
    parser.add_argument('--src_path', type=str, default='./checkpoints/FLARE2022/Training/FLARE22_LabeledCase50/images/')
    parser.add_argument('--seg_path', type=str, default='./checkpoints/FLARE2022/Training/FLARE22_LabeledCase50/labels/')

    parser.add_argument('--out_path', type=str, default='./checkpoints/predata/')
    args = parser.parse_args()
    return args


def load_nii(nii_path):
    tmp_img = sitk.ReadImage(nii_path)
    spacing = tmp_img.GetSpacing()
    spacing = spacing[::-1]
    data_np = sitk.GetArrayFromImage(tmp_img)
    return data_np, tmp_img, spacing

def is_flip(direction):
    x_d = direction[0]; y_d = direction[4]; z_d = direction[8]
    if x_d < 0:
        x_flip = True
    elif x_d > 0:
        x_flip = False
    else:
        raise ValueError(f" wrong x direction {x_d} in sitk img!")
    if y_d < 0:
        y_flip = True
    elif y_d > 0:
        y_flip = False
    else:
        raise ValueError(f" wrong y direction {y_d} in sitk img!")
    if z_d < 0:
        z_flip = True
    elif z_d > 0:
        z_flip = False
    else:
        raise ValueError(f" wrong z direction {z_d} in sitk img!")
    return x_flip, y_flip, z_flip

def gen_single_data(info):
    
    try:
        pid, vol_file, out_dir, seg_file, sitk_lock = info
        save_file = os.path.join(out_dir, f'{pid}.npz')
        sitk_lock.acquire()
        vol, sitk_img, spacing_vol = load_nii(vol_file)
        sitk_lock.release()

        if  not os.path.exists(seg_file):
            return None
        seg, _, _ = load_nii(seg_file)
        seg = seg.astype(np.uint8)
              

        x_flip, y_flip, z_flip = is_flip(sitk_img.GetDirection())
        print(pid, sitk_img.GetDirection(), x_flip, y_flip, z_flip)
        if x_flip:
            vol = np.ascontiguousarray(np.flip(vol, 2))
            seg = np.ascontiguousarray(np.flip(seg, 2)).astype(np.uint8)
        if y_flip:
            vol = np.ascontiguousarray(np.flip(vol, 1))
            seg = np.ascontiguousarray(np.flip(seg, 1)).astype(np.uint8)
        if z_flip:
            vol = np.ascontiguousarray(np.flip(vol, 0))
            seg = np.ascontiguousarray(np.flip(seg, 0)).astype(np.uint8)
        
        vol_shape = np.array(vol.shape)
        seg_shape = np.array(seg.shape)
        if np.any(vol_shape != seg_shape):
            print('pid vol shape != seg shape: ', pid)
            return None

        np.savez_compressed(
            save_file,
            vol=vol,
            seg=seg,
            src_spacing=np.array(spacing_vol),
        )
        print(f'{pid} successed')
        return save_file
    except:
        traceback.print_exc()
        sitk_lock.release()
        print(f'{pid} failed')
        return None


def write_list(request, result):
    write_queue.put(result)

def list_save(data_list, out):
    with open(out, 'w') as f:
        for data in data_list:
            f.writelines(data + '\r\n')

def gen_lst(out_dir):
    train_save_list = os.path.join(out_dir, 'train.lst')
    val_save_list = os.path.join(out_dir, 'val.lst')

    data_list = sorted([p for p in os.listdir(out_dir) if p.endswith('.npz')])
    train_list = random.sample(data_list, int(len(data_list) * 0.8))
    val_list = sorted(list(set(data_list) - set(train_list)))
    print('num of train data and val data: ', len(train_list), len(val_list))
    list_save(train_list, train_save_list)
    list_save(val_list, val_save_list)

if __name__ == '__main__':
    sitk_lock = threading.RLock()
    write_queue = Queue()
    args = parse_args()
    src_dir = args.src_path
    out_dir = args.out_path
    seg_dir = args.seg_path
    os.makedirs(out_dir, exist_ok=True)
    data_lst = []
    for pid in sorted(os.listdir(src_dir)):
        vol_file = os.path.join(src_dir, pid)
        seg_file = os.path.join(seg_dir, pid.replace('_0000.nii.gz', '.nii.gz')) 2022
        info = [pid.replace('.nii.gz', ''), vol_file, out_dir, seg_file, sitk_lock]
        data_lst.append(info)
    pool = threadpool.ThreadPool(30)
    requests = threadpool.makeRequests(gen_single_data, data_lst, write_list)
    ret_lines = [pool.putRequest(req) for req in requests]
    pool.wait()

    print(f'finshed {len(data_lst)} patient.')
    gen_lst(out_dir)
    
