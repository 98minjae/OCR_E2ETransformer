from matcher import HungarianMatcher
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import torch
import editdistance
import numpy as np

def get_accuracy(outputs, targets):
  
  matcher = HungarianMatcher()
  indices = matcher(outputs,targets)
  # num_boxes = sum(len(t["labels"]) for t in targets)
  # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
  # if is_dist_avail_and_initialized():
  #     torch.distributed.all_reduce(num_boxes)
  # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()  
  
  if indices[0][0].size()[0] == 0:       #indices가 빈 텐서인 경우. 즉 매칭된 query가 존재하지 않을때

    print("No matched indices") 
    
  else: 
    out = outputs['pred_text']   # n_q,27,2500   #각각의 query별로 27개의 seq 각각에 2500개의 class에 대한 softmax
    target = targets[0]['transcript'] #  1, n_t, 27
    #print('out',out,out.size())
    n_samples = target.size(0)
    end_idxs = [t == 1 for t in target]
    end_idxs = [i.nonzero() for i in end_idxs]
    #print(end_idxs)
    

    pred_idxs = indices[0][0]
    target_idxs = indices[0][1]
    #print('pred_idx',pred_idxs)
    #print('target_idxs',target_idxs)

    

    total_correct_char = 0
    num_char = sum(end_idxs)
    
    ed = 0

    for i, j in zip(pred_idxs, target_idxs):

      pred_seq = torch.argmax(out[i], dim = 1)# dim 확인
      p = pred_seq == 1
      target_seq = target[j]  # dim 확인
      
      correct_char = pred_seq == target_seq
      correct_char = correct_char[:end_idxs[j]]
      total_correct_char += correct_char.sum()

      ed += editdistance.eval(pred_seq,target_seq)/end_idxs[j]

    
    # a = (total_correct_char, num_char)
    # b = (ed, n_samples)
    #print(a,b)
    return total_correct_char, num_char, ed, n_samples



 