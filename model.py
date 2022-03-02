import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from util import box_ops
from backbone import build_backbone
from matcher import build_matcher
from transformer import build_transformer
from loss import SetCriterion
from vitstr import create_vitstr

from backbone import build_backbone
from matcher import build_matcher
from transformer import build_transformer

from roi_rotate import ROIRotate
from util.upsampling import Upsampling 

class Model(nn.Module):
    def __init__(self,backbone,transformer,num_classes,num_queries,vitstr,vocab_size):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        
        #recog  branch
        self.roi_rotate = ROIRotate()
        self.upsampling = Upsampling()
        self.conv1 = nn.Conv2d(hidden_dim,2048,1)
        self.vitstr = vitstr
        
        self.fc1 = nn.Linear(vocab_size,2048)
        self.fc2 = nn.Linear(2048,4098)
        self.fc3 = nn.Linear(4098,vocab_size)
        #weight initialization
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.input_proj.weight)
        nn.init.kaiming_normal_(self.class_embed.weight)

    def forward(self,samples):
        #origin = samples.tensors
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
        
        outputs_class = self.class_embed(hs)   
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        # print('outputs_coord:', outputs_coord)
        # print('outputs_coord:', outputs_coord.size()) #(6, 1, 100, 4)
      
        # print('outputs_class:', outputs_class)
        # print('outputs_class:', outputs_class.size()) #(6, 1, 100, 2)

        # recognition
        f = []
        for _, feature in enumerate(features):
          src,_ = feature.decompose()
          f.append(src)
    
        weight = 0.8
        f[-1] = weight * self.conv1(memory) + (1-weight) * f[-1] # 원본이 많이 상하면 vitstr성능이 떨어질까봐 그니까 objcet가 있는 곳 마저도 흐려질수도 있으니깐!!@
        f = f[::-1].copy()
        restored_features = self.upsampling(f) # resnet의 각 layer의 output의 정볼 모아서 upsampling 했어
        
        possible_region_idx = torch.argmax(outputs_class[-1], dim = 2) # 어떻게 이걸 매칭할까 고민했는데 우리는 이상황에서는 target을 모르니까 이런식으로 접근했어
        possible_region = [outputs_coord[-1][0][i] for i,j in enumerate(possible_region_idx[0]) if j==1]
        
        #print('outputs_coord : ', outputs_coord.size())
        #print('possible_region_idx[0] : ', possible_region_idx[0])
        #print('outputs_coord[-1][0][1]: ', outputs_coord[-1][0][1])

        predicted_num_boxes = len(possible_region) 
        
        #print('number of queries : ',predicted_num_boxes)
        if predicted_num_boxes == 0:  

          pred_class = outputs_class[-1]
          pred_coord = outputs_coord[-1]
          detected = False
          
          out = {'pred_logits': pred_class, 'pred_boxes': pred_coord}
          return detected, out
        
        else:
          _, _, h, w = restored_features.size()
          scale_factor = torch.tensor([w,h]).repeat(2).to('cuda')

          one_to_batched_for_recog = []
          for i in range(predicted_num_boxes):
              box = box_ops.box_cxcywh_to_xyxy(possible_region[i])
              box = box * scale_factor
              one_to_batched_for_recog.append([box])
        
          recog_input = self.roi_rotate(restored_features, one_to_batched_for_recog, predicted_num_boxes)
          recog_input = torch.sum(recog_input, dim = 1).unsqueeze(1) / 32 # 그냥 ㅎ 
        
          pred_text = self.vitstr(recog_input)
          pred_text = self.fc1(pred_text) #fc layer 추가
          pred_text = self.fc2(pred_text)
          pred_text = self.fc3(pred_text)

          idx = torch.ones(100).to('cuda')
        
          pred_class = outputs_class[-1][idx == possible_region_idx].unsqueeze(0)
          pred_coord = outputs_coord[-1][idx == possible_region_idx].unsqueeze(0)
              
          detected = True
          out = {'pred_logits': pred_class, 'pred_boxes': pred_coord, 'pred_text':pred_text}
          return detected, out

def build_model(args):
    num_classes = 2
    device = torch.device(args.device)
    
    backbone = build_backbone(args)

    transformer = build_transformer(args)

    vitstr = create_vitstr(num_tokens=args.vocab_size, model="vitstr_tiny_patch16_224")

    model = Model(
        backbone,
        transformer,
        num_classes = 2,
        num_queries = args.num_queries,
        vitstr = vitstr,
        vocab_size = args.vocab_size
    )
    ####
    if args.resume != '' :
      print("Load pretrained DETR")
      checkpoint = torch.hub.load_state_dict_from_url(args.resume,map_location='cpu',check_hash =True)
      model.load_state_dict({k[len("detr."):]: v for k,v  in checkpoint['model'].items() if "detr." in k and not('class_embed' or 'bbox') in k},strict = False)
      print('finish loading DETR')
    ###
    matcher = build_matcher(args)
    weight_dict = {'loss_ce':args.cls_loss_coef, 'loss_bbox':args.bbox_loss_coef,'loss_giou':args.giou_loss_coef,'loss_recog':args.recog_loss_coef}

    losses = ['labels', 'boxes', 'cardinality','text']
    
    criterion = SetCriterion(num_classes, matcher = matcher, weight_dict = weight_dict, eos_coef = args.eos_coef, losses = losses,character = args.character)
    criterion.to(device)
    postprocessors = {'bbox':PostProcess()}

    return model, criterion, postprocessors
    
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x