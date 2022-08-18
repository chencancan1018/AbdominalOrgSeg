import argparse
import os
import sys
import cc3d
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm


try:
    from predictor import SegModel, SegPredictor
except Exception:
    raise

def parse_args():
    parser = argparse.ArgumentParser(description='Test for abdomen_seg_mask3d')

    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--input_path', default='./inputs/', type=str)
    parser.add_argument('--output_path', default='./outputs/', type=str)

    parser.add_argument('--model_file_global', default=' ', type=str)
    parser.add_argument('--network_file_global', type=str, default=' ',)

    parser.add_argument('--model_file_caseLG', default=' ', type=str)
    parser.add_argument('--network_file_caseLG', type=str, default=' ',)

    parser.add_argument('--model_file_casegall', default=' ', type=str)
    parser.add_argument('--network_file_casegall', type=str, default=' ',)
    
    parser.add_argument('--model_file_caselsk', default=' ', type=str)
    parser.add_argument('--network_file_caselsk', type=str, default=' ',)

    parser.add_argument('--model_file_casepdsto', default=' ', type=str)
    parser.add_argument('--network_file_casepdsto', type=str, default=' ',)

    parser.add_argument('--model_file_caseAIVC', default=' ', type=str)
    parser.add_argument('--network_file_caseAIVC', type=str, default=' ',)

    parser.add_argument('--model_file_caseEso', default=' ', type=str)
    parser.add_argument('--network_file_caseEso', type=str, default=' ',)

    parser.add_argument('--model_file_caseAG', default=' ', type=str)
    parser.add_argument('--network_file_caseAG', type=str, default=' ',)

    parser.add_argument('--model_file_caseRAG4', default=' ', type=str)
    parser.add_argument('--network_file_caseRAG4', type=str, default=' ',)

    parser.add_argument('--model_file_caseLAG', default=' ', type=str)
    parser.add_argument('--network_file_caseLAG', type=str, default=' ',)


    args = parser.parse_args()
    return args

def inference(predictor: SegPredictor, hu_volume, spacing, sup_mask=None):
    pred_array = predictor.forward(hu_volume, spacing, sup_mask=sup_mask)
    return pred_array

def single_connected_region(mask, values, all_labels=None, threshold=0):
    if all_labels is None:
        all_labels = values 
    out = np.zeros(mask.shape, dtype=np.uint8)
    for v in values:
        temp = mask.copy()
        temp[temp != v] = 0
        if v in list(np.unique(temp)):
            labeled, N = cc3d.connected_components((temp > 0), return_N=True)
            area = np.sum(labeled == 1); target = 1
            if N >= 2:
                for idx in range(2, N+1):
                    if np.sum(labeled == idx) > area:
                        target = idx
                        area = np.sum(labeled == idx)
            assert (target > 0)
            out[labeled == target] = v

    if len(values) != len(all_labels):
        res_labels = sorted(list(set(all_labels) - set(values)))
        for v in res_labels:
            if threshold > 0:
                temp = mask.copy()
                temp[temp != v] = 0
                labeled, N = cc3d.connected_components((temp > 0), return_N=True)
                for idx in range(1, N+1):
                    if np.sum(labeled == idx) > threshold:
                        out[labeled == idx] = v
            else:
                out[mask == v] = v
    return out

def single_connected_region_pdsto(mask, values):
    out = np.zeros(mask.shape, dtype=np.uint8)
    for v in [1,2]:
        temp = mask.copy()
        temp[temp != v] = 0
        if v in list(np.unique(temp)):
            labeled, N = cc3d.connected_components((temp > 0), return_N=True)
            area = np.sum(labeled == 1); target = 1
            if N >= 2:
                for idx in range(2, N+1):
                    if np.sum(labeled == idx) > area:
                        target = idx
                        area = np.sum(labeled == idx)
            assert (target > 0)
            out[labeled == target] = v
    v = 3
    temp = mask.copy()
    temp[temp != 3] = 0
    if v in list(np.unique(temp)):
        labeled, N = cc3d.connected_components((temp > 0), return_N=True)
        for idx in range(1, N+1):
            if np.sum(labeled == idx) > 10000:
                target = idx
                area = np.sum(labeled == idx)
                out[labeled == idx] = v
    return out


def find_valid_region(mask, values, low_margin=[0,0,0], up_margin=[0,0,0], MaxConnectedRegion=False):
    if MaxConnectedRegion:
        mask = single_connected_region(mask, values)

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

def is_flip(direction):
    x_d = direction[0]; y_d = direction[4]; z_d = direction[8]
    if int(x_d) < 0:
        x_flip = True
    elif int(x_d) > 0:
        x_flip = False
    else:
        raise ValueError(f" wrong x direction {x_d} in sitk img!")
    if int(y_d) < 0:
        y_flip = True
    elif int(y_d) > 0:
        y_flip = False
    else:
        raise ValueError(f" wrong y direction {y_d} in sitk img!")
    if int(z_d) < 0:
        z_flip = True
    elif int(z_d) > 0:
        z_flip = False
    else:
        raise ValueError(f" wrong z direction {z_d} in sitk img!")
    return x_flip, y_flip, z_flip


def main(input_path, output_path, gpu, args):

    model_segmask_global = SegModel(
        model_f=args.model_file_global,
        network_f=args.network_file_global,
    ) 
    model_segmask_caseLG = SegModel(
        model_f=args.model_file_caseLG,
        network_f=args.network_file_caseLG,
    )    
    model_segmask_casegall = SegModel(
        model_f=args.model_file_casegall,
        network_f=args.network_file_casegall,
    )  
    model_segmask_caselsk = SegModel(
        model_f=args.model_file_caselsk,
        network_f=args.network_file_caselsk,
    )
    model_segmask_casepdsto = SegModel(
        model_f=args.model_file_casepdsto,
        network_f=args.network_file_casepdsto,
    )
    model_segmask_caseAIVC = SegModel(
        model_f=args.model_file_caseAIVC,
        network_f=args.network_file_caseAIVC,
    )
    model_segmask_caseEso = SegModel(
        model_f=args.model_file_caseEso,
        network_f=args.network_file_caseEso,
    )
    model_segmask_caseAG = SegModel(
        model_f=args.model_file_caseAG,
        network_f=args.network_file_caseAG,
    )
    model_segmask_caseRAG4 = SegModel(
        model_f=args.model_file_caseRAG4,
        network_f=args.network_file_caseRAG4,
    )
    model_segmask_caseLAG = SegModel(
        model_f=args.model_file_caseLAG,
        network_f=args.network_file_caseLAG,
    )

    
    predictor_segmask_global = SegPredictor(
        gpu = gpu,
        model = model_segmask_global,
    )
    predictor_segmask_caseLG = SegPredictor(
        gpu = gpu,
        model = model_segmask_caseLG,
    )
    predictor_segmask_casegall = SegPredictor(
        gpu = gpu,
        model = model_segmask_casegall,
    )
    predictor_segmask_caselsk = SegPredictor(
        gpu = gpu,
        model = model_segmask_caselsk,
    )
    predictor_segmask_casepdsto = SegPredictor(
        gpu = gpu,
        model = model_segmask_casepdsto,
    )
    predictor_segmask_caseAIVC = SegPredictor(
        gpu = gpu,
        model = model_segmask_caseAIVC,
    )
    predictor_segmask_caseEso = SegPredictor(
        gpu = gpu,
        model = model_segmask_caseEso,
    )
    predictor_segmask_caseAG = SegPredictor(
        gpu = gpu,
        model = model_segmask_caseAG,
    )
    predictor_segmask_caseRAG4 = SegPredictor(
        gpu = gpu,
        model = model_segmask_caseRAG4,
    )
    predictor_segmask_caseLAG = SegPredictor(
        gpu = gpu,
        model = model_segmask_caseLAG,
    )

    os.makedirs(output_path, exist_ok=True)

    pids = sorted(os.listdir(input_path))
    for pid in tqdm(pids):
        print(pid)
        print('data load ......') 

        sitk_img= sitk.ReadImage(os.path.join(input_path, pid))
        hu_volume = sitk.GetArrayFromImage(sitk_img)
        src_shape = hu_volume.shape
        spacing = sitk_img.GetSpacing()
        spacing = spacing[::-1]

        x_flip, y_flip, z_flip = is_flip(sitk_img.GetDirection())
        if x_flip:
            hu_volume = np.ascontiguousarray(np.flip(hu_volume, 2))
        if y_flip:
            hu_volume = np.ascontiguousarray(np.flip(hu_volume, 1))
        if z_flip:
            hu_volume = np.ascontiguousarray(np.flip(hu_volume, 0))

        global_seg = inference(predictor_segmask_global, hu_volume, spacing)
        global_seg_shape = global_seg.shape
        heatmap = np.zeros(global_seg_shape, dtype=np.uint8)

        # for liver, spleen and kidney
        pmin, pmax = find_valid_region(global_seg.copy(), [1,2,3,13], low_margin=[4, 20, 20], up_margin=[4, 20, 20])
        vol_case = hu_volume[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] 
        pred_case = inference(predictor_segmask_caselsk, vol_case, spacing)  
        pred_case = single_connected_region(pred_case, [1,2,3,4])
        temp = np.zeros(global_seg_shape, dtype=np.uint8)
        temp[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] = pred_case
        heatmap[temp == 1] = 1; heatmap[temp == 3] = 3; heatmap[temp == 4] = 13
        temp_heat = heatmap.copy()
        temp_heat[temp == 2] = 2
        
        # for pancreas, duodenum and stomach
        pmin, pmax = find_valid_region(global_seg.copy(), [4, 11, 12], low_margin=[4, 20, 20], up_margin=[4, 20, 20])
        vol_case = hu_volume[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]]
        pred_case = inference(predictor_segmask_casepdsto, vol_case, spacing)
        pred_case = single_connected_region_pdsto(pred_case, [1,2,3])
        temp = np.zeros(global_seg_shape, dtype=np.uint8)
        temp[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] = pred_case
        heatmap[temp == 1] = 4
        heatmap[temp == 2] = 11
        # heatmap[temp == 3] = 12
        temp_heat[temp == 3] = 12

        temp_eso = np.zeros(global_seg_shape, dtype=np.uint8)
        temp_eso[temp == 2] = 11
        temp_eso[global_seg == 10] = 10

        # for aorta and IVC 
        pmin, pmax = find_valid_region(global_seg.copy(), [5, 6], low_margin=[4, 20, 20], up_margin=[4, 20, 20], MaxConnectedRegion=True)
        vol_case = hu_volume[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]]  
        pred_case = inference(predictor_segmask_caseAIVC, vol_case, spacing)
        pred_case = single_connected_region(pred_case, [1,2], all_labels=[1,2], threshold=300)
        temp = np.zeros(global_seg_shape, dtype=np.uint8)
        temp[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] = pred_case
        temp_heat[temp == 1] = 5
        temp_heat[temp == 2] = 6

        # for esopha
        # pmin, pmax = find_valid_region(temp_eso.copy(), [10], low_margin=[10, 30, 30], up_margin=[10, 30, 30], MaxConnectedRegion=True)
        pmin, pmax = find_valid_region(temp_eso.copy(), [10], low_margin=[20, 30, 30], up_margin=[20, 30, 30], MaxConnectedRegion=True)
        if pmin is None:
            print("no esopha")
        else:
            vol_case = hu_volume[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]]  
            pred_case = inference(predictor_segmask_caseEso, vol_case, spacing)
            pred_case = single_connected_region(pred_case, [1])
            temp = np.zeros(global_seg_shape, dtype=np.uint8)
            temp[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] = pred_case
            heatmap[heatmap == 10] = 0; heatmap[temp == 1] = 10

        # for RAG
        pmin, pmax = find_valid_region(global_seg.copy(), [7], low_margin=[15, 40, 40], up_margin=[15, 40, 40], MaxConnectedRegion=True)
        if (pmin is None) and (pmax is None):
            print('no RAG')
        else:
            vol_case = hu_volume[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] 
            pred_case = inference(predictor_segmask_caseAG, vol_case, spacing) 
            pred_case = single_connected_region(pred_case, [1])
            temp = np.zeros(global_seg_shape, dtype=np.uint8)
            temp[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] = pred_case

            pmin, pmax = find_valid_region(temp.copy(), [1], low_margin=[2, 20, 20], up_margin=[2, 20, 20])
            vol_case = hu_volume[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] 
            pred_case = inference(predictor_segmask_caseRAG4, vol_case, spacing)
            pred_case = single_connected_region(pred_case, [1])
            temp = np.zeros(global_seg_shape, dtype=np.uint8)
            temp[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] = pred_case
            heatmap[temp == 1] = 7

        # for LAG
        pmin, pmax = find_valid_region(global_seg.copy(), [8], low_margin=[10, 40, 40], up_margin=[10, 40, 40], MaxConnectedRegion=True)
        if (pmin is None) and (pmax is None):
            print('no LAG')
        else:
            vol_case = hu_volume[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] 
            pred_case = inference(predictor_segmask_caseAG, vol_case, spacing) 
            temp = np.zeros(global_seg_shape, dtype=np.uint8)
            temp[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] = pred_case
            heatmap[temp == 1] = 8

            pmin, pmax = find_valid_region(heatmap.copy(), [8], low_margin=[2, 20, 20], up_margin=[2, 20, 20])
            vol_case = hu_volume[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] 
            pred_case = inference(predictor_segmask_caseLAG, vol_case, spacing) 
            pred_case = single_connected_region(pred_case, [1])
            temp = np.zeros(global_seg_shape, dtype=np.uint8)
            temp[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] = pred_case
            heatmap[heatmap == 8] = 0; heatmap[temp == 1] = 8

        # for Gallbladder
        pmin, pmax = find_valid_region(global_seg.copy(), [1, 9], low_margin=[4, 40, 40], up_margin=[4, 40, 40])
        vol_case = hu_volume[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] 
        pred_case = inference(predictor_segmask_caseLG, vol_case, spacing)  
        temp = np.zeros(global_seg_shape, dtype=np.uint8)
        temp[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] = pred_case

        pmin, pmax = find_valid_region(temp.copy(), [2], low_margin=[4, 20, 20], up_margin=[4, 20, 20])
        # pmin, pmax = find_valid_region(heatmap.copy(), [9], low_margin=[2, 20, 20], up_margin=[2, 20, 20])
        if (pmin is None) and (pmax is None):
            print('no gall')
        else:
            vol_case = hu_volume[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] 
            pred_case = inference(predictor_segmask_casegall, vol_case, spacing)
            labeled, N = cc3d.connected_components((pred_case.copy() > 0), return_N=True)
            if N >= 2:
                for idx in range(1, N+1):
                    if np.sum(labeled==idx) < 1500:
                        labeled[labeled == idx] = 0
            labeled = (labeled > 0).astype(np.uint8)
            temp = np.zeros(global_seg_shape, dtype=np.uint8)
            temp[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] = labeled
            heatmap[heatmap == 9] = 0; heatmap[temp == 1] = 9
        
        heatmap[temp_heat == 2] = 2
        heatmap[temp_heat == 12] = 12
        heatmap[temp_heat == 5] = 5
        heatmap[temp_heat == 6] = 6


        # for remove galls' invalid region
        pmin, pmax = find_valid_region(heatmap.copy(), [9], low_margin=[2, 10, 10], up_margin=[2, 10, 10])
        if (pmin is not None) and (pmax is not None):
            gall_mask = heatmap[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]].copy()
            labeled, N = cc3d.connected_components((gall_mask == 9), return_N=True)
            if N >= 2:
                for idx in range(1, N+1):
                    if np.sum(labeled==idx) < 1500:
                        labeled[labeled == idx] = 0
            labeled = (labeled > 0).astype(np.uint8)
            temp = np.zeros(global_seg_shape, dtype=np.uint8)
            temp[pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] = labeled
            heatmap[heatmap == 9] = 0; heatmap[temp > 0] = 9

        if x_flip:
            heatmap = np.ascontiguousarray(np.flip(heatmap, 2))
        if y_flip:
            heatmap = np.ascontiguousarray(np.flip(heatmap, 1))
        if z_flip:
            heatmap = np.ascontiguousarray(np.flip(heatmap, 0))

        print('Check output classes: ', np.unique(heatmap))        
        temp = heatmap.copy()
        assert tuple(temp.shape) == tuple(src_shape), f"{pid} has wrong heatmap shape!"
        segments_itk = sitk.GetImageFromArray(temp.astype(np.uint8))
        segments_itk.CopyInformation(sitk_img)
        sitk.WriteImage(segments_itk, os.path.join(output_path, pid))

if __name__ == '__main__':
    args = parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        gpu=args.gpu,
        args=args,
    )
