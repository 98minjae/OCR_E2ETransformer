import torch.nn as nn
import torch
import torch.nn.functional as F


from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from convert import TokenLabelConverter

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, character):
        super().__init__()
        self.num_classes = 2
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.converter = TokenLabelConverter(character)

        self.recog_cls_weight = torch.ones(len(character)+2).to('cuda')
        self.recog_cls_weight[0] = 3
    def forward(self,outputs,targets,recognition = True):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size. 1 dict= 1 img
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        if recognition:
          for loss in self.losses:
              losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes,recognition)) 
        else:
          loss_lst_wo_recog = list(self.losses)
          loss_lst_wo_recog.remove('text')
          for loss in loss_lst_wo_recog:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes,recognition))
          
        return losses
    

    def loss_recog(self,outputs,targets,indices,num_boxes):
        # outputs: list of dicts. {predclass :(bs, query, 2), :(bs, query, 4), (bs, reduced query, 27, 2500)}
        # targets: batch * (list of dicts). (boxes: (num of boxes, 4 coordinates),'labels':(num of boxes, 1), 'transcript': (num of boxes,27))
        
        if indices[0][0].size()[0] == 0:  # for No matched query
          losses = {'loss_recog': 0}
          return losses

        else:
          seq_len = targets[0]['transcript'].size()[1]          
          out = outputs['pred_text']     
          target = targets[0]['transcript']

          entire_loss = 0

          pred_idxs = indices[0][0]
          target_idxs = indices[0][1]

          for i, j in zip(pred_idxs, target_idxs):
            pred_seq = out[i]  
            target_seq = target[j]  
            
            entire_loss += F.cross_entropy(pred_seq, target_seq,self.recog_cls_weight)  

          losses = {'loss_recog': torch.divide(entire_loss, len(indices))}
          
          return losses

    def loss_labels(self,outputs,targets,indices,num_boxes,log = True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
 
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses 

   



    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def get_loss(self, loss, outputs, targets, indices, num_boxes,recognition=True, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,      
        }
        if recognition:
          loss_map['text'] = self.loss_recog
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    





