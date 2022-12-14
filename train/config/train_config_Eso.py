
# Distributed
gpus=4
total_epoch=90
distributed=True
deterministic = False

# DataLoader
batch_size=8
shuffle=True
num_workers=4

# Paras for Dataset
import numpy as np
dataset="LagSegDataset"
win_level = [50, 50]
win_width = [400, 200]
in_ch = len(win_level)
patch_size = [64, 224, 224]
threshold = 0.96
train=dict(
    dst_list_file='./checkpoints/predata/train/train.lst',
    data_root='./checkpoints/predata/train',
    patch_size=patch_size,
    sample_frequent=12,
    win_level=win_level,
    win_width=win_width,
    aug_paras={
        "prob": 0.5,
        "patch_size": patch_size,
        "roi_scale": 0.5,
        "max_roi_scale": 1.5,
        "rot_range_x": (-np.pi/9, np.pi/9),
        "rot_range_y": (-np.pi/9, np.pi/9),
        "rot_range_z": (-np.pi/9, np.pi/9),
        "rot_90": True,
        "flip": False,
        "bright_bias": (-0.4, 0.4),
        "bright_weight": (-0.4, 0.4),
        "translate_x": (-5.0, 5.0),
        "translate_y": (-5.0, 5.0),
        "translate_z": (-5.0, 5.0),
        "scale_x": (-0.2, 0.2),
        "scale_y": (-0.2, 0.2),
        "scale_z": (-0.2, 0.2),
        "elastic_sigma_range": (3, 5),  # x,y,z
        "elastic_magnitude_range": (100, 200),
    }
),
val=dict(
    dst_list_file='./checkpoints/predata/val/val.lst',
    data_root='./checkpoints/predata/val',
    patch_size=patch_size,
    sample_frequent=1,
    win_level=win_level,
    win_width=win_width,
    aug_paras=[],
),

# Model
model=dict(
    in_ch=in_ch, 
    channels=32, 
    blocks=3, 
    use_aspp=False, 
    is_aux=True,
    head_type="sig",
    classes=1,
    apply_sync_batchnorm=True,
)

# Optimizer and LR
optimizer = dict(lr=1e-4, weight_decay=1e-5)
optimizer_config={}
lr_config=dict(step=[30, 60], gamma=0.2)

# Log
is_logger=True

# Mixed Precsion
fp16=False

# Validation
validate=False

# Pretrain
load_from=None
save_dir='./checkpoints/results/resunet_lag'

