# CUDA_VISIBLE_DEVICES=0,1 python3.7 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main.py --config ./config/train_config_AG.py

# CUDA_VISIBLE_DEVICES=0,1 python3.7 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main.py --config ./config/train_config_AIVC.py

CUDA_VISIBLE_DEVICES=2,3 python3.7 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main.py --config ./config/train_config_coarse.py

# CUDA_VISIBLE_DEVICES=0,1 python3.7 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main.py --config ./config/train_config_Eso.py

# CUDA_VISIBLE_DEVICES=0,1 python3.7 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main.py --config ./config/train_config_Gall.py

# CUDA_VISIBLE_DEVICES=0,1 python3.7 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main.py --config ./config/train_config_Lag.py

# CUDA_VISIBLE_DEVICES=0,1 python3.7 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main.py --config ./config/train_config_LG.py

# CUDA_VISIBLE_DEVICES=0,1 python3.7 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main.py --config ./config/train_config_LSK.py

# CUDA_VISIBLE_DEVICES=0,1 python3.7 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main.py --config ./config/train_config_PDSto.py

# CUDA_VISIBLE_DEVICES=0,1 python3.7 -m torch.distributed.launch --master_port 29501 --nproc_per_node=2 main.py --config ./config/train_config_Rag.py