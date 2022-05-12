import random
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from config import get_parse
from _train_ import train
from model import build_model

import datasets
import util.misc as utils
from datasets import build_dataset 
import torch.utils.data
import torchvision
import wandb



def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model,criterion, postprocessors= build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
      model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
      model_without_ddp = model.module

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }]
    optimizer = torch.optim.AdamW(param_dicts,lr=args.lr,weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # init wandb
    NAME= "cyclic lr test"
    wandb.init( project="e2eocr_experiments", entity="yai_e2eocr1", name = NAME)
    wandb.define_metric("epoch")
    wandb.config.update(args)
    

    print('finish init optim, scheduler')

    # data
    dataset_train = build_dataset(image_set='train', args=args)  
    dataset_val = build_dataset(image_set='test', args=args)
    # dataset_test = build_dataset(image_set='test', args=args)
    
    print('finish building dataset')

  
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)


    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=2)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=2)

    
    #train
    max_val_acc = 0
    for epoch in range(args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train(
            model,criterion, data_loader_train, optimizer, device, epoch, data_loader_val, args.val_epochs, NAME, args.clip_max_norm, max_val_acc)
        lr_scheduler.step()


if __name__ == '__main__':
    args = get_parse()
    main(args)
