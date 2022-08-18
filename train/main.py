import os
import sys
import datetime
import shutil
import argparse
import numpy as np
from mmcv import Config
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast
from collect_env import get_pretty_env_info

from data.match import dataset_match
from utils.utils import get_logger
from models.network import SegNetwork, SegSigHead, SegSoftHead
from models.backbone import ResUnet

class Trainner(object):
    def __init__(self, config, gpu=None):

        self.config = config
        os.makedirs(self.config.save_dir, exist_ok=True)
        

        if self.config.distributed:
            rank = self.config.get("rank", None)
            if rank is None:
                rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1
                num_gpus = torch.cuda.device_count()
                torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(backend='nccl', rank=rank)

        self.model = self._get_model()

        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if self.config.is_logger:
            self.logger = get_logger(self.config.save_dir)

        if self.config.distributed:
            ngpus_per_node = torch.cuda.device_count()
            
            self.model.cuda() 
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[torch.cuda.current_device()])
            self.batch_size = int(self.config.batch_size)
            self.num_workers = int(self.config.num_workers)
            if dist.get_rank():
                self.logger.info("System platform: ".format(sys.platform))
                self.logger.info("PyTorch compiling details: \n{}".format(get_pretty_env_info()))
                self.logger.info("CUDA version: {}".format(torch.version.cuda))
                self.logger.info("CUDA available: {}".format(torch.cuda.is_available))
                self.logger.info("If distributed: ".format(self.config.distributed))
                self.logger.info('Total epochs: {}'.format(self.config.total_epoch)) 
                self.logger.info('Patch size: {}'.format(self.config.patch_size)) 
                self.logger.info('Preprocess parallels batch_size: {}'.format(self.batch_size)) 
                self.logger.info('Preprocess parallels num_workers: {}'.format(self.num_workers)) 
                self.logger.info('save_dir: {}'.format(self.config.save_dir)) 
                self.logger.info('Model Architecture: {}'.format(self.model)) 
        else:
            raise NotImplementedError("Only DistributedDataParallel is supported.")

        self.lr = self.config.optimizer["lr"]
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=self.config.optimizer["weight_decay"]
        )
        

        self.start_epoch = 0
        if self.config.load_from is not None:
            self._load_checkpoint(self.config.load_from, gpu)

    def write_logger(self, information):
        if dist.get_rank():
            self.logger.info(information)
    
    def run(self):
        SegDataset = dataset_match(self.config.dataset)
        train_dataset = SegDataset(self.config.train[0])
        if self.config.validate:
            val_dataset = SegDataset(self.config.val[0])

        if self.config.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            if self.config.validate:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        else:
            train_sampler = None
            if self.config.validate:
                val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.num_workers,
            sampler=train_sampler,
            drop_last=True,
        )
        if self.config.validate:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=val_sampler,
                drop_last=True,
            )
            self.write_logger('val samples per epoch: {}'.format(len(val_loader)))
        self.write_logger('train samples per epoch: {}'.format(len(train_loader)))
       

        total_epoch = self.config.total_epoch
        for epoch in range(self.start_epoch, total_epoch):
            if self.config.distributed:
                train_sampler.set_epoch(epoch)
                if self.config.validate:
                    val_sampler.set_epoch(epoch)
            start_time = datetime.datetime.now()
            self.train(train_loader, epoch)
            is_best = False; val_loss = 1e+3
            if self.config.validate:
                temp = self.val(val_loader, epoch)
                if val_loss > temp:
                    val_loss = temp
                    is_best = True

            self.write_logger('End of epoch {}, time: {}'.format(epoch, datetime.datetime.now() - start_time))
            self._save_checkpoint(self.config.save_dir, epoch, is_best=False)
        self.write_logger("Finish training!")
            
    
    def train(self, train_loader, epoch):
        self.model.train()
        self.lr = self._get_lr(epoch, self.lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr
        for idx, data_batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            fp16 = self.config.get("fp16")
            if fp16:
                with autocast():
                    losses = self.model(**data_batch)
            else:
                losses = self.model(**data_batch)

            loss, log_vars = self._parse_loss(losses)
            info = list()
            info.append(datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
            info.append(epoch); info.append(idx); info.append(len(train_loader)); info.append(self.lr)
            for k, v in log_vars.items():
                info.append(v)
            self.write_logger(
                "{} - Epoch: [{}][{}/{}], lr: {}, loss_pos: {:.5f}, loss_neg: {:.5f}, dice loss: {:.5f}".format(*info)
            )
            loss.backward()
            self.optimizer.step()

    def val(self, val_loader, epoch):
        self.model.eval()
        valloss = []
        for idx, data_batch in enumerate(val_loader):
            fp16 = self.config.get("fp16")
            if fp16:
                with autocast():
                    losses = self.model(**data_batch)
            else:
                losses = self.model(**data_batch)
            loss, log_vars = self._parse_loss(losses)
            info = list()
            info.append(datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
            info.append(epoch); info.append(idx); info.append(len(val_loader)); info.append(self.lr)
            for k, v in log_vars.items():
                info.append(v)
            valloss.append(loss.detach().cpu().numpy())
        valloss = np.mean(valloss)
        self.write_logger(
                "{} - Epoch: [{}], val loss: {:.5f}".format(datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S"), epoch, valloss)
            )
        return valloss

    def _parse_loss(self, losses):
        log_vars = OrderedDict()
        for k, v in losses.items():
            log_vars[k] = v.mean()

        loss = sum(v for k, v in log_vars.items() if 'loss' in k)
        for k, v in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                v = v.data.clone()
                dist.all_reduce(v.div_(dist.get_world_size()))
            log_vars[k] = v.item()
        return loss, log_vars

    def _get_model(self):

        backbone = ResUnet(
            self.config.model["in_ch"], 
            channels = self.config.model["channels"],
            blocks = self.config.model["blocks"],
            use_aspp = self.config.model["use_aspp"], 
            is_aux = self.config.model["is_aux"],
        )
        if self.config.model["head_type"] == "sig":
            head = SegSigHead(in_channels=self.config.model["channels"], classes=1)
        elif self.config.model["head_type"] == "soft":
            head = SegSoftHead(in_channels=self.config.model["channels"], classes=self.config.model["classes"])
        else:
            raise KeyError("wrong network head!")
        return SegNetwork(
            backbone, head,
            apply_sync_batchnorm=self.config.model["apply_sync_batchnorm"]
        )

    def _get_lr(self, epoch, base_lr):

        lr_config = self.config.lr_config
        step = lr_config["step"] 
        gamma = lr_config["gamma"]
        if isinstance(step, int):
            return base_lr * (gamma**(epoch // step))
        
        exp = len(step)
        for i, s in enumerate(step):
            if epoch < s:
                exp = i
                break
        return base_lr * gamma**exp

    def _load_checkpoint(self, path, gpu):

        print("Loading pre_trained model from: {}".format(path))
        self.write_logger("Loading pre_trained model from: {}".format(path))
        if gpu is None:
            checkpoint = torch.load(path)
        else:
            loc = "cuda:{}".format(gpu)
            checkpoint = torch.load(path, map_location=loc)
        self.start_epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr = checkpoint['lr']
    
    def _save_checkpoint(self, path, epoch, is_best=False):

        model_path = os.path.join(path, "epoch_{}.pth".format(str(epoch)))
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer_dict": self.optimizer.state_dict(),
            "lr": self.lr
        }
        torch.save(checkpoint, model_path)
        if is_best:
            shutil.copyfile(model_path, os.path.join(path, "model_best.pth"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train for abdominal_organ_seg_mask3d')
    parser.add_argument('--config', default='./config/train_config_coarse.py', type=str)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    config = Config.fromfile(args.config)
    trainner = Trainner(config)
    trainner.run()