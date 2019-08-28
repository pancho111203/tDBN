import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchplus.tools import change_default_args
from torchplus.nn import Empty

import sparseconvnet as scn

def init_sparse_tensor(features, coords, shape, dims=3):
    coords = coords.to(dtype=torch.long)
    ret = scn.InputBatch(dims, shape)

    for i in range(0, len(features)):
        ret.add_sample()
        ret.set_locations(coords[i], features[i]) 
    return ret

class PointGridExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=1,
                 name='PointGridExtractor',
                 use_norm=True,
                 use_dropout=False,
                 average_pool=False,
                 feature_sizes=[64, 128, 256, 384, 512],
                 version=0,
                 device=None,
                 **kwargs):
        '''
        Versions: 
            0: default version, every layer's filter is reduced from nxnxnx to nx1x1 -> 1xnx1 -> 1x1xn
            1: version with last 2 layers changed, using 3x3x3->3x3x3 each one
        '''
        super(PointGridExtractor, self).__init__()
        self.name = name
        self.num_input_features = num_input_features
        self.output_shape = output_shape
        self.use_norm = use_norm
        self.device = device
        # same as voxel_generator.grid_size[::-1] + [1, 0, 0]
        self.sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        if average_pool is False:
            Pool = change_default_args(dimension=3)(scn.MaxPooling)
        else:
            Pool = change_default_args(dimension=3)(scn.AveragePooling)

        self.middle_conv = scn.Sequential()
        # 1x216x7992x7056
        self.middle_conv.add(self.smconv_block(num_input_features, feature_sizes[0], (7, 7, 7)))
        self.middle_conv.add(self.smconv_block(feature_sizes[0], feature_sizes[0], (7, 7, 7)))
        self.middle_conv.add(Pool(pool_size=(3, 3, 3), pool_stride=(3, 3, 3)))
        # 64x72x2664x2352

        self.middle_conv.add(self.smconv_block(feature_sizes[0], feature_sizes[1], (7, 7, 7)))
        self.middle_conv.add(self.smconv_block(feature_sizes[1], feature_sizes[1], (7, 7, 7)))
        self.middle_conv.add(Pool(pool_size=(3, 3, 3), pool_stride=(3, 3, 3)))
        # 128x24x888x784

        self.middle_conv.add(self.smconv_block(feature_sizes[1], feature_sizes[2], (5, 5, 5)))
        self.middle_conv.add(self.smconv_block(feature_sizes[2], feature_sizes[2], (5, 5, 5)))
        self.middle_conv.add(Pool(pool_size=(3, 2, 2), pool_stride=(3, 2, 2)))
        # 256x8x444x392

        # if version == 0:
        #     self.middle_conv.add(self.smconv_block(feature_sizes[2], feature_sizes[3], (5, 1, 1)))
        #     self.middle_conv.add(self.smconv_block(feature_sizes[3], feature_sizes[3], (1, 5, 1)))
        #     self.middle_conv.add(self.smconv_block(feature_sizes[3], feature_sizes[3], (1, 1, 5)))

        #     self.middle_conv.add(Pool(pool_size=(2, 2, 2), pool_stride=(2, 2, 2)))
        #     # 384x4x222x196

        #     self.middle_conv.add(self.smconv_block(feature_sizes[3], feature_sizes[4], (5, 1, 1)))
        #     self.middle_conv.add(self.smconv_block(feature_sizes[4], feature_sizes[4], (1, 5, 1)))
        #     self.middle_conv.add(self.smconv_block(feature_sizes[4], feature_sizes[4], (1, 1, 5)))

        #     self.middle_conv.add(Pool(pool_size=(2, 2, 2), pool_stride=(2, 2, 2)))
        #     # 512x2x111x98

        # elif version == 1:
        self.middle_conv.add(self.smconv_block(feature_sizes[2], feature_sizes[3], (3, 3, 3)))
        self.middle_conv.add(self.smconv_block(feature_sizes[3], feature_sizes[3], (3, 3, 3)))
        self.middle_conv.add(Pool(pool_size=(2, 2, 2), pool_stride=(2, 2, 2)))
        # 384x4x222x196

        self.middle_conv.add(self.smconv_block(feature_sizes[3], feature_sizes[4], (3, 3, 3)))
        self.middle_conv.add(self.smconv_block(feature_sizes[4], feature_sizes[4], (3, 3, 3)))
        self.middle_conv.add(Pool(pool_size=(2, 2, 2), pool_stride=(2, 2, 2)))
        # 512x2x111x98
        # else:
        #     raise Exception('Invalid version: {}'.format(version))

        # To DENSE
        self.middle_conv.add(scn.SparseToDense(3, feature_sizes[4]))

        # TODO make it conditional on device configured
        # self.middle_conv.cuda()

        spatial_size = self.middle_conv.input_spatial_size(torch.LongTensor([2, 111, 98]))
        print('Calculated spatial size: {}, type: {}'.format(spatial_size, type(spatial_size)))
        self.input_layer = scn.InputLayer(dimension=3, spatial_size=spatial_size, mode=2)

        # TODO apply dense conv to resulting features?
        # if its too much memory i could remove z axis applying pooling to it and compute normal CNN on 2d matrix

    def smconv_block(self, in_f, out_f, kernel):
        block = scn.Sequential()
        block.add(scn.SubmanifoldConvolution(3, in_f, out_f, kernel, not self.use_norm))
        if self.use_norm:
            block.add(scn.BatchNormReLU(out_f))
        return block

    def forward(self, voxel_features, coors, batch_size):
        self.input_layer.to(device=self.device)
        coors = coors.cpu()
        # voxel_features: [B*N, f] contains all point features concatenated
        # coordinates: [B*N, 4] on column 0 contains batch index, on the rest contains point coordinates

        # swap batch_index to 4th column

        coors = torch.index_select(coors, 1, torch.LongTensor([1,2,3,0])).long()
        voxel_features = voxel_features.reshape((voxel_features.shape[0], self.num_input_features))

        ret = self.input_layer((coors, voxel_features, batch_size))

        ret = self.middle_conv(ret)

        # print('Middle return shape: {}'.format(ret.shape))

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret