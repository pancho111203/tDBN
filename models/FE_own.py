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

import operator
import torch
import warnings
from torch.nn.parallel.data_parallel import *


def debug(txt):
    print(txt)

class Pyramid(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 name='Pyramid',
                 use_residual=True,
                 blocks = [(32, 15), (64, 11), (96, 7), (128, 5), (160, 3)], # define blocks, the tuple represent (num_filters, kernel). (blocks are divided by one downsample op)
                 layers_per_block = 2, # each layer consists of 2 sscnn if use residual is true, and 1 if false
                 downsample_type='max_pool2',
                 leakiness=0,
                 dense_blocks=[(192, 3, (2, 1, 1)), (224, 3, (2, 1, 1)), (256, 3, (2, 1, 1))], # define final dense blocks, with (num_filters, kernel, stride)
                 out_filters=512,
                 final_z_dim=12, # TODO URG obtain from dense block output
                 **kwargs):
        super(Pyramid, self).__init__()
        self.name = name
        self.use_residual = use_residual
        self.layers_per_block = layers_per_block
        self.blocks = blocks
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        if downsample_type == 'max_pool2':
            Downsample = change_default_args(dimension=3, pool_size=(2, 2, 2), pool_stride=(2, 2, 2))(scn.MaxPooling)
        else:
            # scn.Convolution(dimension, num_filter_fpn[k-1], num_filter_fpn[k], 3, 2, False)
            raise ValueError('Invalid downsample type')

        sparse_shape = np.array(output_shape[1:4]) # + [1, 0, 0]   
        
        m = scn.Sequential()
        (num_filters, kernel_size) = blocks[0]

        self.block(m, 1, num_filters, dimension=3, residual_blocks = use_residual, kernel_size=kernel_size)
        for _ in range(layers_per_block-1):
            self.block(m, num_filters, num_filters, dimension=3, residual_blocks = use_residual, kernel_size=kernel_size)

        self.block_models = [m]
        prev_num_filters = num_filters

        for k, (num_filters, kernel_size) in enumerate(blocks[1:]):
            k = k + 1

            m = scn.Sequential()
            # downsample
            m.add(scn.BatchNormLeakyReLU(prev_num_filters ,leakiness=leakiness))
            m.add(Downsample)

            for _ in range(layers_per_block):
                self.block(m, num_filters, num_filters, dimension=3, residual_blocks = use_residual, kernel_size=kernel_size)

            self.block_models.append(m)
            prev_num_filters = num_filters


        # TODO use CNNBLOCK if sparsity is low enough
        self.dense_block = scn.Sequential()
        self.dense_block.add(scn.BatchNormLeakyReLU(prev_num_filters , leakiness=leakiness))
        self.dense_block.add(Downsample)

        for (num_filters, kernel, stride) in range(0, final_scnns):
            self.dense_block.add(scn.Convolution(3, prev_num_filters, num_filters, kernel, stride, bias=False))
            self.dense_block.add(scn.BatchNormLeakyReLU(num_filters, leakiness=leakiness))
            prev_num_filters = num_filters
            # TODO should i keep downsampling here??? check

        self.z_combiner = scn.Sequential()
        self.z_combiner.add(scn.Convolution(3, prev_num_filters, out_filters, (final_z_dim, 1, 1), (1, 1, 1), bias=False))
        self.z_combiner.add(scn.BatchNormLeakyReLU(out_filters, leakiness=leakiness))
        self.z_combiner.add(scn.SparseToDense(3, out_filters)) #TODO should this be 2D instead of 3d??


    # NOTE: this blocks start with BatchNorm+ReLu and end without
    def block(self, m, a, b, dimension=3, residual_blocks=False, leakiness=0, kernel_size=3):  # default using residual_block
        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, kernel_size, False))
                    .add(scn.BatchNormLeakyReLU(b,leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, kernel_size, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.BatchNormLeakyReLU(a,leakiness=leakiness))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, kernel_size, False)))

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()[:, [1, 2, 3, 0]]
        ret = self.scn_input((coors.cpu(), voxel_features, batch_size))

        # TODO output % sparsity and dimensions after every layer to see if changes need to be made

        debug('Input:')
        debug(ret)

        for k, model in enumerate(self.block_models):
            ret = model(ret)
            debug('{} block output:'.format(k))
            debug(ret)

        ret = self.dense_block(ret)

        debug('Dense Block output:')
        debug(ret)

        ret = self.z_combiner(ret)

        N, C, D, H, W = ret.shape
        debug('Return Shape: {}'.format(ret.shape))
        
        output = ret.view(N, C*D, H, W)
        print('TODO: ensure that the feature map is of a good size for the RCNN')
        print('Size: {}'.format(output.shape))
        return output
