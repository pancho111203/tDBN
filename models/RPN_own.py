
import time
from enum import Enum
from functools import reduce

import numpy as np
import sparseconvnet as scn
import torch
from torch import nn
from torch.nn import functional as F

import torchplus
from torchplus import metrics
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
from tDBN.core import box_torch_ops
from tDBN.core.losses import (WeightedSigmoidClassificationLoss,
                              WeightedSmoothL1LocalizationLoss,
                              WeightedSoftmaxClassificationLoss)
import operator
import torch
import warnings



class Direct(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 3, 3, 3],
                 layer_strides=[2, 2, 2, 2],
                 num_filters=[ 64, 128, 256, 512 ],
                 upsample_strides=[1, 2, 4, 4],
                 num_upsample_filters= [ 64, 128, 256, 256, 448 ],
                 num_input_filters=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 name='Direct',
                 output_channels = 384, # TODO channels gotten from feature extractor
                 **kwargs):
        super(Direct, self).__init__()
        self.name = name
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        assert len(layer_nums) == 4
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d( output_channels, num_cls, 1)
        self.conv_box = nn.Conv2d( output_channels , num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d( output_channels, num_anchor_per_loc * 2, 1)


    def forward(self, x, bev=None):
        out = x #TODO for now, we process the input from feature extractor directly (see how this goes)

        box_preds = self.conv_box(out)
        cls_preds = self.conv_cls(out)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(out)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict



