# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
from pycocotools import mask as coco_mask
import os
import os.path as osp


import datasets.transforms as T

import numpy as np
import typing
import pathlib
from PIL import Image

# from .text_transforms import Transform
# from ..utils.util import str_label_converter
# from .text_datautils import check_and_validate_polys, normalize_iamge, str_label_converter
from . import text_utils as data_utils

from convert import TokenLabelConverter

import cv2


class TextDetection(Dataset):
    def __init__(self, data_root, transforms, character, return_masks:bool=False, training:bool=True):
        data_root = pathlib.Path(data_root)
        
        self.images_root = data_root / 'imgs' #SynthText/imgs
        self.gt_root = data_root / 'gt' #SynthText/gt
        self.training = training
        self.transform = make_coco_transforms('train') 
        
        self.images, self.bboxs, self.transcripts, self.ids = self.__loadGT() #SynthText/imgs 파일 전체가 loadGT함수로 들어가므로 배치단위가 아닌 전체 데이터셋 단위로 봐야할듯

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(character,return_masks) 


    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")

        image_id = self.ids[idx]
        bbox = self.bboxs[idx]
        transcript = self.transcripts[idx]

        target = {'image_id': image_id, 'annotations': bbox, 'transcript': transcript}
        #print('target[annotations]:', target['annotations'])
        
        img, target = self.prepare(img, target)
        # print(target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target,img_path

    def __loadGT(self):
        all_bboxs = []
        all_texts = []
        all_images = []
        all_ids = []
        check =0
        
        for i_idx, image in enumerate(self.images_root.glob('*.jpg')): #현재 디렉토리의 모든 jpg 파일단위, 즉 반환하는 리스트가 배치단위는 아닌듯하다
            # image = pathlib.Path('/data/ocr/det/icdar2015/detection/train/imgs/img_756.jpg')
            # gt = pathlib.Path('/data/ocr/det/icdar2015/detection/train/gt/gt_img_756.txt')
            gt = self.gt_root / image.with_name('gt_{}'.format(image.stem)).with_suffix('.txt').name
            
            if os.path.isfile(gt) == False :
                print('[file error]',check,gt)
                check+=1
            else :
                with gt.open(mode='r') as f: #한 이미지내에 존재하는 query들
                    
                    bboxes = []
                    texts = []
                    for line in f:
                        text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',') #공백이랑 에러문자열들 제거하고 ,로 text 구분
                        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
                        #bbox = [[x1, y1], [x3, y3]]
                        bbox = [[min(x1,x2,x3,x4),min(y1,y2,y3,y4)],[max(x1,x2,x3,x4),max(y1,y2,y3,y4)]]
                        transcript = text[8]
                        if transcript == '###' and self.training:
                            continue
                        bboxes.append({"image_id":i_idx, "category_id":1, "bbox":bbox}) #label 1 for text, 0 for background
                        texts.append(transcript)

                    if len(bboxes) > 0: #현재 파일내에 존재하는 모든 jpg파일중 gt존재하면 list에 정리해서 넣어주자
                        bboxes = np.array(bboxes)
                        all_bboxs.append(bboxes)
                        all_texts.append(texts)
                        all_images.append(image)
                        all_ids.append(i_idx)

        return all_images, all_bboxs, all_texts, all_ids

    def __len__(self):
        return len(self.images)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self,character, return_masks=False):
        self.return_masks = return_masks
        
        self.converter = TokenLabelConverter(character)

    def __call__(self, image, target):
        # target need "image_id" & "annotations" 
        # in annotation, bbounding box info dict list 
        # in bounding box object : id, image_id, category_id, bbox, segmentation, keypoints, num_keypoints, score, area, iscrowd  + transcript
        # text need image_id, category_id(label), bbox
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        ##Transcript
        transcript =[[obj] for obj in target["transcript"]]
        
        #print('before converter  ',transcript)
        transcript = self.converter.encode(transcript)
        #print('after converter  ', transcript)
        transcript = transcript.clone().detach()
    
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        # obj -> dict : id, image_id, category_id, bbox, segmentation, keypoints, num_keypoints, score, area, iscrowd

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4) # every bbox in one image

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        transcript =transcript[keep]
 
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["transcript"] = transcript

        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        # area = torch.tensor([obj["area"] for obj in anno])
        # iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        area = torch.tensor([0 for obj in anno])
        iscrowd = torch.tensor([0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        

        # target info : "image id", "annotations", "boxes", "labels", 
        # "masks"(if return_mask true), "image_id", "keypoints", "area", "iscrowd", "orig_size", "size"
        
        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales =[224,256] #[480, 512]# 

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales),
            #T.RandomSelect(
               # T.RandomResize(scales, max_size=1333),
                #T.Compose([
                #    T.RandomResize([224,256]),
                    #T.RandomSizeCrop(224),
                    #T.RandomResize(scales, max_size=1333),
                #])
        #]),
            normalize
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize(scales),
            normalize
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    data_root = Path(args.data_path)
    assert data_root.exists(), f'provided COCO path {root} does not exist'
    
    #dataset = TextDetection(data_root, transforms=make_coco_transforms(image_set),character = args.character)
    print('data_path: ',osp.join(data_root,image_set))

    if image_set != 'test' :
      dataset = TextDetection(data_root=osp.join(data_root,image_set), transforms=make_coco_transforms(image_set),character = args.character)
    else :  #test일 때  transforms val mode
      dataset = TextDetection(data_root=osp.join(data_root,image_set), transforms=make_coco_transforms('val'),character = args.character)

    return dataset
