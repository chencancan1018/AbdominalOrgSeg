# AbdominalOrgSeg
## Introduction
- AbdominalOrgSeg is an open source, PyTorch-based segmentation method for 3D medical image. 
- The full code and paper is to be updated.


## Installation
#### Environment
- Ubuntu 18.04
- Python 3.7+
- Pytorch 1.7.1+
- CUDA 10.2+ 

1.Git clone
```
git clone 
```

2.Install dependencies
```
pip install -r requirements.txt
```

## Get Started
### preprocessing
1. Download [FLARE22](https://flare22.grand-challenge.org/Dataset/).
2. Copy image and mask to './checkpoints/FLARE2022/' folder.
4. Run the data preprocess with the following command:
```bash
cd train/
python generate_data.py
```

## Training
- Edit the "train/config/XXXXXXXXXXXXX.py"
- Edit the "train/dist_train.sh"
- Run all models' training by the following command 
```bash
cd train/
sh dist_train.sh
```

## Inference
- Copy "train/checkpoint/results/XXXXXX" to "infer/model/"
- Copy "train/config/XXXXXXXX.py" to "infer/model/"
- Run the inference by the following command
```bash
cd infer/
sh predict.sh
```

## Acknowledgement
Thanks for FLARE organizers.
