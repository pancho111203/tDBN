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

# TODO create visualization of sparsity map (showing sparsity changes per layer)
def debug(txt):
    # print(txt)
    pass

def get_sparsity(tensor):
    # if isinstance(tensor, scn.SparseConvNetTensor):
    #     sl = tensor.get_spatial_locations()
    #     non_zero = float((sl[:, 3] == 0).sum()) # only from 1st element of batch
    #     total = float(reduce(lambda x, y: x * y, tensor.spatial_size))
    #     print('non_zero: {}, total: {}, sparsity: {:.10f}'.format(non_zero, total, non_zero/total))
    # elif isinstance(tensor, torch.Tensor):
    #     non_zero = float(len(torch.nonzero(tensor[0].sum(dim=0)))) # only from 1st element of batch
    #     total = float(reduce(lambda x, y: x * y, tensor.shape[2:]))
    #     print('non_zero: {}, total: {}, sparsity: {:.10f}'.format(non_zero, total, non_zero/total))
    # else:
    #     print('Cant get sparsity for type: {}'.format(type(tensor)))
    pass
    
class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class Pyramid(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=False,
                 name='Pyramid',
                 use_residual=True,
                 blocks = [(64, 15), (96, 11), (128, 7), (192, 5)], # define blocks, the tuple represent (num_filters, kernel). (blocks are divided by one downsample op)
                 layers_per_block = 2, # each layer consists of 2 sscnn if use residual is true, and 1 if false
                 downsample_type='max_pool2',
                 leakiness=0,
                 dense_blocks=[(224, (3, 3, 3), (2, 1, 1)), (256, (3, 3, 3), (2, 1, 1)), (384, (3, 3, 3), (2, 1, 1))], # define final dense blocks, with (num_filters, kernel, stride)
                #  out_filters=512,
                #  final_z_dim=12,
                dense_bias=False,
                 **kwargs):
        super(Pyramid, self).__init__()
        self.name = name
        self.use_residual = use_residual
        self.layers_per_block = layers_per_block
        self.blocks = blocks
        use_norm = False # TODO this is invalidating config passed...
        if use_norm:
            BatchNorm3d = nn.BatchNorm3d
        else:
            BatchNorm3d = Empty

        if downsample_type == 'max_pool2':
            Downsample = change_default_args(dimension=3, pool_size=(2, 2, 2), pool_stride=(2, 2, 2))(scn.MaxPooling)
        else:
            # scn.Convolution(dimension, num_filter_fpn[k-1], num_filter_fpn[k], 3, 2, False)
            raise ValueError('Invalid downsample type')

        sparse_shape = np.array(output_shape[1:4]) # + [1, 0, 0]   
        
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape

        m = scn.Sequential()
        (num_filters, kernel_size) = blocks[0]

        self.block(m, 1, num_filters, dimension=3, residual_blocks = use_residual, kernel_size=kernel_size, use_batch_norm=False)
        for _ in range(layers_per_block-1):
            self.block(m, num_filters, num_filters, dimension=3, residual_blocks = use_residual, kernel_size=kernel_size, use_batch_norm=use_norm)

        self.block_models = [m]
        prev_num_filters = num_filters

        for k, (num_filters, kernel_size) in enumerate(blocks[1:]):
            k = k + 1

            m = scn.Sequential()
            # downsample
            m.add(scn.BatchNormLeakyReLU(prev_num_filters ,leakiness=leakiness))
            m.add(Downsample())
            
            self.block(m, prev_num_filters, num_filters, dimension=3, residual_blocks = use_residual, kernel_size=kernel_size, use_batch_norm=use_norm)
            for _ in range(layers_per_block-1):
                self.block(m, num_filters, num_filters, dimension=3, residual_blocks = use_residual, kernel_size=kernel_size, use_batch_norm=use_norm)

            self.block_models.append(m)
            prev_num_filters = num_filters

        self.block_models.append(scn.Sequential().add(scn.BatchNormLeakyReLU(prev_num_filters, leakiness=leakiness)).add(Downsample()))
        self.block_models = ListModule(*self.block_models)

        # this version uses CNNBLOCK if sparsity is low enough
        self.sparse_to_dense = scn.SparseToDense(3, prev_num_filters)

        self.dense_block = []
        for (num_filters, kernel, stride) in dense_blocks:
            pad = tuple(((np.array(list(kernel)) - 1) / 2).astype(np.int))
            m = nn.Sequential(
                nn.Conv3d(prev_num_filters, num_filters, kernel, stride, padding=pad, bias=dense_bias),
                BatchNorm3d(num_filters),
                nn.LeakyReLU(negative_slope=leakiness)
            )
            self.dense_block.append(m)
            prev_num_filters = num_filters

        self.dense_block = ListModule(*self.dense_block)

        # ## this version used SCNN instead of normal CNN
        # self.dense_block = scn.Sequential()
        # self.dense_block.add(scn.BatchNormLeakyReLU(prev_num_filters , leakiness=leakiness))
        # self.dense_block.add(Downsample())

        # for (num_filters, kernel, stride) in dense_blocks:
        #     self.dense_block.add(scn.Convolution(3, prev_num_filters, num_filters, kernel, stride, bias=False))
        #     self.dense_block.add(scn.BatchNormLeakyReLU(num_filters, leakiness=leakiness))
        #     prev_num_filters = num_filters
        #     # TODO should i keep downsampling here??? check

        # self.z_combiner = scn.Sequential()
        # self.z_combiner.add(scn.Convolution(3, prev_num_filters, out_filters, (final_z_dim, 1, 1), (1, 1, 1), bias=False))
        # self.z_combiner.add(scn.BatchNormLeakyReLU(out_filters, leakiness=leakiness))
        # self.z_combiner.add(scn.SparseToDense(3, out_filters)) #TODO should this be 2D instead of 3d??


    # NOTE: this blocks start with BatchNorm+ReLu and end without
    def block(self, m, a, b, dimension=3, residual_blocks=False, leakiness=0, kernel_size=3, use_batch_norm=True):  # default using residual_block
        if use_batch_norm:
            Activation = lambda channels: scn.BatchNormLeakyReLU(channels,leakiness=leakiness)
        else:
            Activation = lambda channels: scn.LeakyReLU(leakiness)

        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(Activation(a))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, kernel_size, False))
                    .add(Activation(b))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, kernel_size, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(Activation(a))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, kernel_size, False)))

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()[:, [1, 2, 3, 0]]
        ret = self.scn_input((coors.cpu(), voxel_features, batch_size))

        debug('Input:')
        debug(ret)
        get_sparsity(ret)

        for k, model in enumerate(self.block_models):
            ret = model(ret)
            debug('{} block output:'.format(k))
            debug(ret)
            get_sparsity(ret)
        # out: [  8, 200, 176]

        # ret = self.dense_block(ret)

        # debug('Dense Block output:')
        # debug(ret)

        # ret = self.z_combiner(ret)

        ret = self.sparse_to_dense(ret)
        debug('Dense return shape: {}'.format(ret.shape))
        get_sparsity(ret)

        for k, model in enumerate(self.dense_block):
            ret = model(ret)
            debug('Dense block {} return shape: {}'.format(k, ret.shape))
            get_sparsity(ret)

        # TODO global maxpool just in case it's not 1

        N, C, D, H, W = ret.shape      
        output = ret.view(N, C*D, H, W)
        #TODO ensure that the feature map is of a good size for the RCNN'
        debug('Size: {}'.format(output.shape))
        return output




class Pyramid_Light(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 name='Pyramid_Light',
                 use_residual=True,
                 blocks = [(64, 15), (80, 11), (96, 7), (128, 5)], # define blocks, the tuple represent (num_filters, kernel). (blocks are divided by one downsample op)
                 layers_per_block = 2, # each layer consists of 2 sscnn if use residual is true, and 1 if false
                 downsample_type='max_pool2',
                 leakiness=0,
                #  dense_blocks=[(160, (3, 3, 3), (2, 1, 1)), (192, (3, 3, 3), (2, 1, 1)), (224, (3, 3, 3), (2, 1, 1))], # define final dense blocks, with (num_filters, kernel, stride)
                #  out_filters=512,
                #  final_z_dim=12,
                 **kwargs):
        super(Pyramid_Light, self).__init__()
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
        
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape

        m = scn.Sequential()
        (num_filters, kernel_size) = blocks[0]

        self.block(m, 1, num_filters, dimension=3, residual_blocks = use_residual, kernel_size=kernel_size, use_batch_norm=False)
        for _ in range(layers_per_block-1):
            self.block(m, num_filters, num_filters, dimension=3, residual_blocks = use_residual, kernel_size=kernel_size)

        self.block_models = [m]
        prev_num_filters = num_filters

        for k, (num_filters, kernel_size) in enumerate(blocks[1:]):
            k = k + 1

            m = scn.Sequential()
            # downsample
            m.add(scn.BatchNormLeakyReLU(prev_num_filters ,leakiness=leakiness))
            m.add(Downsample())
            
            self.block(m, prev_num_filters, num_filters, dimension=3, residual_blocks = use_residual, kernel_size=kernel_size)
            for _ in range(layers_per_block-1):
                self.block(m, num_filters, num_filters, dimension=3, residual_blocks = use_residual, kernel_size=kernel_size)

            self.block_models.append(m)
            prev_num_filters = num_filters

        self.block_models.append(scn.Sequential().add(scn.BatchNormLeakyReLU(prev_num_filters, leakiness=leakiness)).add(Downsample()))
        self.block_models = ListModule(*self.block_models)

        # this version uses CNNBLOCK if sparsity is low enough
        self.sparse_to_dense = scn.SparseToDense(3, prev_num_filters)

        # self.dense_block = []
        # for (num_filters, kernel, stride) in dense_blocks:
        #     pad = tuple(((np.array(list(kernel)) - 1) / 2).astype(np.int))
        #     m = nn.Sequential(
        #         nn.Conv3d(prev_num_filters, num_filters, kernel, stride, padding=pad),
        #         nn.BatchNorm3d(num_filters),
        #         nn.LeakyReLU(negative_slope=leakiness)
        #     )
        #     self.dense_block.append(m)
        #     prev_num_filters = num_filters

        # self.dense_block = ListModule(*self.dense_block)

        # ## this version used SCNN instead of normal CNN
        # self.dense_block = scn.Sequential()
        # self.dense_block.add(scn.BatchNormLeakyReLU(prev_num_filters , leakiness=leakiness))
        # self.dense_block.add(Downsample())

        # for (num_filters, kernel, stride) in dense_blocks:
        #     self.dense_block.add(scn.Convolution(3, prev_num_filters, num_filters, kernel, stride, bias=False))
        #     self.dense_block.add(scn.BatchNormLeakyReLU(num_filters, leakiness=leakiness))
        #     prev_num_filters = num_filters
        #     # TODO should i keep downsampling here??? check

        self.z_combiner = nn.Sequential(
            nn.Conv3d(prev_num_filters, prev_num_filters, (8, 1, 1), groups=prev_num_filters),
            nn.BatchNorm3d(num_filters),
            nn.LeakyReLU(negative_slope=leakiness)
        )
        # [8, 224, 2, 194, 170] -> [8, 224, 194, 170]


    # NOTE: this blocks start with BatchNorm+ReLu and end without
    def block(self, m, a, b, dimension=3, residual_blocks=False, leakiness=0, kernel_size=3, use_batch_norm=True):  # default using residual_block
        if use_batch_norm:
            Activation = lambda channels: scn.BatchNormLeakyReLU(channels,leakiness=leakiness)
        else:
            Activation = lambda channels: scn.LeakyReLU(leakiness)

        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(Activation(a))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, kernel_size, False))
                    .add(Activation(b))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, kernel_size, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(Activation(a))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, kernel_size, False)))

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()[:, [1, 2, 3, 0]]
        ret = self.scn_input((coors.cpu(), voxel_features, batch_size))

        debug('Input:')
        debug(ret)

        for k, model in enumerate(self.block_models):
            ret = model(ret)
            debug('{} block output:'.format(k))
            debug(ret)
        # out: [  8, 200, 176]

        # ret = self.dense_block(ret)

        # debug('Dense Block output:')
        # debug(ret)

        # ret = self.z_combiner(ret)

        ret = self.sparse_to_dense(ret)
        debug('Dense return shape: {}'.format(ret.shape))

        ret = self.z_combiner(ret)
        debug('z-combiner return shape: {}'.format(ret.shape))

        # for k, model in enumerate(self.dense_block):
        #     ret = model(ret)
        #     debug('Dense block {} return shape: {}'.format(k, ret.shape))

        # TODO global maxpool just in case it's not 1

        N, C, D, H, W = ret.shape      
        output = ret.view(N, C*D, H, W)
        #TODO ensure that the feature map is of a good size for the RCNN'
        debug('Size: {}'.format(output.shape))
        return output


class Pyramid_LightNoBN(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 name='Pyramid_LightNoBN',
                 use_residual=True,
                 blocks = [(64, 15), (80, 11), (96, 7), (128, 5)], # define blocks, the tuple represent (num_filters, kernel). (blocks are divided by one downsample op)
                 layers_per_block = 2, # each layer consists of 2 sscnn if use residual is true, and 1 if false
                 downsample_type='max_pool2',
                 leakiness=0,
                #  dense_blocks=[(160, (3, 3, 3), (2, 1, 1)), (192, (3, 3, 3), (2, 1, 1)), (224, (3, 3, 3), (2, 1, 1))], # define final dense blocks, with (num_filters, kernel, stride)
                #  out_filters=512,
                #  final_z_dim=12,
                 **kwargs):
        super(Pyramid_LightNoBN, self).__init__()
        self.name = name
        self.use_residual = use_residual
        self.layers_per_block = layers_per_block
        self.blocks = blocks
        if downsample_type == 'max_pool2':
            Downsample = change_default_args(dimension=3, pool_size=(2, 2, 2), pool_stride=(2, 2, 2))(scn.MaxPooling)
        else:
            # scn.Convolution(dimension, num_filter_fpn[k-1], num_filter_fpn[k], 3, 2, False)
            raise ValueError('Invalid downsample type')

        sparse_shape = np.array(output_shape[1:4]) # + [1, 0, 0]   
        
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape

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
            m.add(scn.LeakyReLU(leakiness))
            m.add(Downsample())
            
            self.block(m, prev_num_filters, num_filters, dimension=3, residual_blocks = use_residual, kernel_size=kernel_size)
            for _ in range(layers_per_block-1):
                self.block(m, num_filters, num_filters, dimension=3, residual_blocks = use_residual, kernel_size=kernel_size)

            self.block_models.append(m)
            prev_num_filters = num_filters

        self.block_models.append(scn.Sequential().add(scn.LeakyReLU(leakiness)).add(Downsample()))
        self.block_models = ListModule(*self.block_models)

        self.sparse_to_dense = scn.SparseToDense(3, prev_num_filters)

        self.z_combiner = nn.Sequential(
            nn.Conv3d(prev_num_filters, prev_num_filters, (8, 1, 1), groups=prev_num_filters),
            # nn.BatchNorm3d(num_filters),
            nn.LeakyReLU(negative_slope=leakiness)
        )
        # [8, 224, 2, 194, 170] -> [8, 224, 194, 170]


    # NOTE: this blocks start with BatchNorm+ReLu and end without
    def block(self, m, a, b, dimension=3, residual_blocks=False, leakiness=0, kernel_size=3):  # default using residual_block
        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.LeakyReLU(leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, kernel_size, False))
                    .add(scn.LeakyReLU(leakiness))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, kernel_size, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.LeakyReLU(leakiness))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, kernel_size, False)))

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()[:, [1, 2, 3, 0]]
        ret = self.scn_input((coors.cpu(), voxel_features, batch_size))

        debug('Input:')
        debug(ret)

        for k, model in enumerate(self.block_models):
            ret = model(ret)
            debug('{} block output:'.format(k))
            debug(ret)
        # out: [  8, 200, 176]

        # ret = self.dense_block(ret)

        # debug('Dense Block output:')
        # debug(ret)

        # ret = self.z_combiner(ret)

        ret = self.sparse_to_dense(ret)
        debug('Dense return shape: {}'.format(ret.shape))

        ret = self.z_combiner(ret)
        debug('z-combiner return shape: {}'.format(ret.shape))

        # for k, model in enumerate(self.dense_block):
        #     ret = model(ret)
        #     debug('Dense block {} return shape: {}'.format(k, ret.shape))

        # TODO global maxpool just in case it's not 1

        N, C, D, H, W = ret.shape      
        output = ret.view(N, C*D, H, W)
        #TODO ensure that the feature map is of a good size for the RCNN'
        debug('Size: {}'.format(output.shape))
        return output
