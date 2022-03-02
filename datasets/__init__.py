# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .text import build as build_text


def build_dataset(image_set, args):
    if args.dataset_file == 'text':

        return build_text(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
