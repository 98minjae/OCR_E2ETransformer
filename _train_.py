import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils

import wandb
from util.visualize_results import *
from util.metric import get_accuracy

import random
import time




def train(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, 
                    val_data_loader: Iterable, val_epoch: int, model_name:str, max_norm: float = 0, max_val_acc=0):

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    # init for log
    pass_cnt = 0
    total_img_num = len(data_loader)
    log_dict = {'train/'+k:torch.zeros([1]).to(torch.device('cuda:0')) for k in criterion.weight_dict.keys()} #for log
    log_dict['train/losses'] = torch.zeros([1]).to(torch.device('cuda:0'))

    # acc       
    t_correct_char=0
    t_num_char = 0 
    t_ed = 0 
    t_n_samples = 0

    
    for samples, targets,img_path in metric_logger.log_every(data_loader, print_freq, header):  # what is the targets' output?
        
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] #list(dict{label:좌표})
        detected, outputs = model(samples)

        if not detected:
          pass_cnt+=1

        # else:
          ##viz 추가, img생성 여기에선 안함
          # viz(targets,outputs,img_path[0].as_posix(),epoch,img_viz=False)


        loss_dict = criterion(outputs,targets,recognition = detected)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        

        # update log
        log_dict.update({'train/'+k:log_dict['train/'+k]+loss_dict[k] for k in loss_dict.keys() if k in weight_dict}) #update log dict
        log_dict['train/losses'] += losses

        # acc
        if detected:
          correct_char, num_char, ed, n_samples = get_accuracy(outputs,targets)
          t_correct_char += correct_char.item()
          t_num_char += num_char.item()
          t_ed += ed.item()
          t_n_samples += n_samples
        

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    ########### wandb log : pass_num, losses, lr, acc(to do), image_visualize(to do)
    log_dict = {k: log_dict[k]/total_img_num for k in log_dict.keys()}
    log_dict['train/pass_cnt'] = pass_cnt
    log_dict['train/lr'] = optimizer.param_groups[0]["lr"]
    log_dict['train/ed'] = 1-t_ed/t_n_samples
    log_dict['train/chr_acc'] = t_correct_char / t_num_char

    wandb.log(log_dict)
    
    ########### wandb log image : viz_img(targets, pred, img_path)
    for i in range(2):
      # r_int = random.randint(0,total_img_num-1)
      r_sample, r_target, r_img_path = next(iter(data_loader))
      r_sample = r_sample.to(device)
      r_target = [{k: v.to(device) for k, v in t.items()} for t in r_target] #list(dict{label:좌표})
      r_detected, r_outputs = model(r_sample)
      if detected :
        r_viz_img = viz_img(r_target,r_outputs,r_img_path[0].as_posix())
        r_viz_img = wandb.Image(r_viz_img)
        wandb.log({"train_viz_img"+str(i+1) : r_viz_img})

    
    print("Train Pass Count : ", pass_cnt)


    if epoch%val_epoch == 0: 
      
      # print("###################val in###########")

      with torch.no_grad():
        val_loss = 0.0 
       
        # acc       
        t_correct_char=0
        t_num_char = 0 
        t_ed = 0 
        t_n_samples = 0

        val_pass_cnt = 0
        val_log_dict = {'val/'+k:torch.zeros([1]).to(torch.device('cuda:0')) for k in criterion.weight_dict.keys()} #for log
        val_log_dict['val/losses'] = torch.zeros([1]).to(torch.device('cuda:0'))

        for  samples, targets,img_path in val_data_loader: #val 연결 # img_path 추가
          samples = samples.to(device)
          targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
          detected, outputs = model(samples) 
          
          ##### viz 추가
          # viz(targets,outputs,img_path[0].as_posix(),epoch,img_viz=True)
        
          if not detected:
            pass_cnt+=1
          val_loss_dict = criterion(outputs,targets,recognition = detected)
          weight_dict = criterion.weight_dict
          losses = sum(val_loss_dict[k] * weight_dict[k] for k in val_loss_dict.keys() if k in weight_dict)

          # update log
          val_log_dict.update({'val/'+k:val_log_dict['val/'+k]+val_loss_dict[k] for k in val_loss_dict.keys() if k in weight_dict}) #update log dict
          val_log_dict['val/losses'] += losses
          
          # acc
          if detected:
            correct_char, num_char, ed, n_samples = get_accuracy(outputs,targets)
            t_correct_char += correct_char.item()
            t_num_char += num_char.item()
            t_ed += ed.item()
            t_n_samples += n_samples
        
        print(f'character by charcter acc: {t_correct_char / t_num_char}')
        print(f'edit_distance : {1-t_ed/t_n_samples}')
          
        # wandb log
        val_log_dict = {k: val_log_dict[k]/len(val_data_loader) for k in val_log_dict.keys()}
        val_log_dict['val/pass_cnt'] = val_pass_cnt
        val_log_dict['val/ed'] = 1-t_ed/t_n_samples
        val_log_dict['val/chr_acc'] = t_correct_char / t_num_char
        wandb.log(val_log_dict)

        ########### wandb log image : viz_img(targets, pred, img_path)
        for i in range(2):
          # r_int = random.randint(0,total_img_num-1)
          r_sample, r_target, r_img_path = next(iter(val_data_loader))
          r_sample = r_sample.to(device)
          r_target = [{k: v.to(device) for k, v in t.items()} for t in r_target] #list(dict{label:좌표})
          r_detected, r_outputs = model(r_sample)
          if detected :
            r_viz_img = viz_img(r_target,r_outputs,r_img_path[0].as_posix())
            r_viz_img = wandb.Image(r_viz_img)
            wandb.log({"val_viz_img"+str(i+1) : r_viz_img})

        chr_acc = t_correct_char/t_num_char
        if(max_val_acc< chr_acc):
          max_val_acc = chr_acc
          PATH = '../outputs/'+model_name+'_best.pt'
          torch.save(model.state_dict(),PATH)
        
        if(epoch/val_epoch) % 3 ==0:
          PATH = '../outputs/'+model_name+'_epoch'+str(epoch)+'.pt'
          torch.save(model.state_dict(),PATH)
          
        


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, max_val_acc

