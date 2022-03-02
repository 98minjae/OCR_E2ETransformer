import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from metric import get_accuracy


def train(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, 
                    val_data_loader: Iterable, val_epoch: int, max_norm: float = 0):

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    pass_cnt = 0
    for samples, targets,_  in metric_logger.log_every(data_loader, print_freq, header):  # what is the targets' output?
        
        samples = samples.to(device)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] #list(dict{label:좌표})
             
        detected, outputs = model(samples)

        #acc = get_accuracy(outputs,targets) 이거 원래 여기 놓으면 안되는데 설명해줄거랑 조금 수정 필요한 부분있을까봐 여기 써놨고 지우지마 주석 풀지도 말구 ㅎㅎ
        #print('hi')
        #print(acc)
        #  print("detected, output print", detected, outputs)
        if not detected:
          pass_cnt+=1

        loss_dict = criterion(outputs,targets,recognition = detected)
        weight_dict = criterion.weight_dict
          
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        # print('done!')
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        # print(losses_reduced_scaled)
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

    print("Train Pass Count : ", pass_cnt)


    if epoch%val_epoch == 0: 
      # print("###################val in###########")
      with torch.no_grad():
        val_loss = 0.0 
        t_correct_char=0
        t_num_char = 0 
        t_ed = 0 
        t_n_samples = 0
        for samples, targets,_ in val_data_loader: #val 연결
          samples = samples.to(device)
          targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
          detected, outputs = model(samples)

          if not detected:
            continue

          loss_dict = criterion(outputs, targets) # {'labels', 'boxes', 'cardinality','text'}
          weight_dict = criterion.weight_dict

          losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

          # reduce losses over all GPUs for logging purposes
          loss_dict_reduced = utils.reduce_dict(loss_dict)
          loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
          loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                      for k, v in loss_dict_reduced.items() if k in weight_dict}
          losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

          correct_char, num_char, ed, n_samples = get_accuracy(outputs,targets)
          t_correct_char += correct_char.item()
          t_num_char += num_char.item()
          t_ed += ed.item()
          t_n_samples += n_samples
        
        print(f'character by charcter acc: {t_correct_char / t_num_char}')
        print(f'edit_distance : {1-t_ed/t_n_samples}')
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}