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
  
  if indices[0][0].size()[0] == 0:       #for no matched query

    print("No matched indices") 
    
  else: 
    out = outputs['pred_text']   # num of reduced_queries,27,2350+a   #for 27 sequence in each reduced queries, softmax about 2350+a classes
    target = targets[0]['transcript'] # 1, num of targets, 27
    n_samples = target.size(0)
    end_idxs = [t == 1 for t in target]
    end_idxs = [i.nonzero() for i in end_idxs]

    

    pred_idxs = indices[0][0]
    target_idxs = indices[0][1]

    total_correct_char = 0
    num_char = sum(end_idxs)
    
    ed = 0

    for i, j in zip(pred_idxs, target_idxs):

      pred_seq = torch.argmax(out[i], dim = 1)
      p = pred_seq == 1
      target_seq = target[j]
      
      correct_char = pred_seq == target_seq
      correct_char = correct_char[:end_idxs[j]]
      total_correct_char += correct_char.sum()

      ed += editdistance.eval(pred_seq,target_seq)/end_idxs[j]

    return total_correct_char, num_char, ed, n_samples



 
