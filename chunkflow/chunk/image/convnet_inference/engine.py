#!/usr/bin/env python
__doc__ = """
ConvNet Inference of an image chunk
"""

import time
import numpy as np
from tqdm import tqdm
from warnings import warn
from typing import Union

from chunkflow.chunk import Chunk
# from chunkflow.chunk.affinity_map import AffinityMap


class Engine(object):
    """
        Engine
    convnet inference for a whole block. the patches should aligned with
    the block size.
    """
    def __init__(self,
                 convnet_model: str,
                 convnet_weight_path: str,
                 input_patch_size: Union[tuple, list],
                 output_patch_size: Union[tuple, list] = None,
                 num_output_channels: int = 3,
                 output_patch_overlap: Union[tuple, list] = (4, 64, 64),
                 framework: str = 'identity',
                 batch_size: int = 1,
                 bump: str = 'wu',
                 input_size: tuple = None,
                 mask_output_chunk: bool = False,
                 verbose: bool = False):

        if output_patch_size is None:
            output_patch_size = input_patch_size

        assert len(output_patch_overlap) == 3
        assert len(input_patch_size) == 3
        assert len(output_patch_size) == 3

        self.convnet_model = convnet_model
        self.convnet_weight_path = convnet_weight_path
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        self.num_output_channels = num_output_channels
        self.output_patch_overlap = output_patch_overlap
        self.framework = framework
        self.batch_size = batch_size
        self.bump = bump
        self.mask_output_chunk = mask_output_chunk

        self.output_crop_margin_size = tuple((osz-isz)//2 for osz, isz in zip(
                                    input_patch_size, output_patch_size))
        self.input_patch_overlap = tuple((opo + ocms) for opo, ocms in zip(
                                            output_patch_overlap,
                                            self.output_crop_margin_size))

        self.input_stride = tuple(p - o for p, o in
                                  zip(input_patch_size,
                                      self.input_patch_overlap))
        self.output_stride = tuple(p - o for p, o in zip(
                                output_patch_size, self.output_patch_overlap))

        self.input_size = input_size
        self.verbose = verbose

        self.mask_output_chunk = mask_output_chunk

        # allocate a buffer to avoid redundant memory allocation
        self.input_patch_buffer = np.zeros((batch_size, 1, *input_patch_size),
                                           dtype=np.float32)

        self.patch_slices_list = []

        self._prepare_patch_engine()

    def _check_alignment(self):
        if not self.mask_output_chunk:
            if self.verbose:
                print('patches should align with input chunk when ' +
                      'not using output chunk mask.')
            is_align = tuple((i - o) % s == 0 for i, s, o in zip(
                self.input_size, self.input_stride, self.input_patch_overlap))
            # all axis should be aligned
            # the patches should aligned with input size in case
            # we will not mask the output chunk
            assert np.all(is_align)

    def _construct_patch_slices_list(self):
        """
        create the normalization mask and patch bounding box list
        """
        self.patch_slices_list = []
        # the step is the stride, so the end of aligned patch is
        # input_size - patch_overlap
        for iz in tqdm(range(0, self.input_size[0] - self.input_patch_overlap[0], self.input_stride[0]),
                       disable=not self.verbose, desc='ConvNet Inferece: '):
            if iz + self.input_patch_size[0] > self.input_size[0]:
                iz = self.input_size[0] - self.input_patch_size[0]
                assert iz >= 0
            oz = iz + self.output_crop_margin_size[0]
            for iy in range(0, self.input_size[1] - self.input_patch_overlap[1], self.input_stride[1]):
                if iy + self.input_patch_size[1] > self.input_size[1]:
                    iy = self.input_size[1] - self.input_patch_size[1]
                    assert iy >= 0
                oy = iy + self.output_crop_margin_size[1]
                for ix in range(0, self.input_size[2] - self.input_patch_overlap[2], self.input_stride[2]):
                    if ix + self.input_patch_size[2] > self.input_size[2]:
                        ix = self.input_size[2] - self.input_patch_size[2]
                        assert ix >= 0
                    ox = ix + self.output_crop_margin_size[2]
                    input_patch_slice =  (slice(iz, iz + self.input_patch_size[0]),
                                          slice(iy, iy + self.input_patch_size[1]),
                                          slice(ix, ix + self.input_patch_size[2]))
                    output_patch_slice = (slice(oz, oz + self.output_patch_size[0]),
                                          slice(oy, oy + self.output_patch_size[1]),
                                          slice(ox, ox + self.output_patch_size[2]))
                    self.patch_slices_list.append((input_patch_slice, output_patch_slice))

    def _construct_output_chunk_mask(self):
        if not self.mask_output_chunk:
            return

        if self.verbose:
            print('creating output chunk mask...')

        self.output_chunk_mask = np.zeros(self.output_size[-3:], np.float32)
        for _, output_patch_slice in self.patch_slices_list:
            # accumulate weights using the patch mask in RAM
            self.output_chunk_mask[output_patch_slice] += self.patch_engine.mask_numpy
        # normalize weight, so accumulated inference result multiplies

        # this mask will result in 1
        self.output_chunk_mask = 1.0 / self.output_chunk_mask

    def _check_input_size_and_prepare_data(self, input_size):
        """
        if the input size is consistent with old one, reuse the 
        patch offset list and output chunk mask. Otherwise, recompute them.
        """
        if self.input_size == input_size:
            print('reusing existing patch offset list and output chunk mask.')
            assert self.patch_slices_list is not None
        else:
            if self.input_size is not None:
                warn('the input size has changed, using new intput size.')
            self.input_size = input_size
            self.output_size = tuple(isz-2*ocms for isz, ocms in zip(
                            input_size, self.output_crop_margin_size))
            self._construct_patch_slices_list()
            if self.mask_output_chunk:
                self._construct_output_chunk_mask()

    def __call__(self,
                 input_chunk: np.ndarray,
                 output_buffer: np.ndarray = None):
        """
        args:
            input_chunk (Chunk): input chunk with global offset
        """
        assert isinstance(input_chunk, Chunk)

        if output_buffer is None:
            output_buffer = self._create_output_buffer(input_chunk)

        if np.all(input_chunk == 0):
            print('input is all zero, return zero buffer directly')
            return output_buffer
        
        self._check_input_size_and_prepare_data(input_chunk.shape)
        self._check_alignment()

        if np.issubdtype(input_chunk.dtype, np.integer):
            # normalize to 0-1 value range
            input_chunk = input_chunk / np.iinfo(input_chunk.dtype).max

        #if input_chunk.dtype == np.uint8:
        #    input_chunk = input_chunk.astype(np.float32) / 255.0

        if self.verbose:
            chunk_time_start = time.time()

        # iterate the offset list
        for i in tqdm(range(0, len(self.patch_slices_list), self.batch_size),
                      disable=not self.verbose,
                      desc='ConvNet Inference ...'):
            if self.verbose:
                start = time.time()

            batch_slices = self.patch_slices_list[i:i + self.batch_size]
            for batch_idx, slices in enumerate(batch_slices):
                self.input_patch_buffer[batch_idx, 0, :, :, :] = input_chunk[slices[0]]

            if self.verbose:
                end = time.time()
                print('prepare %d input patches takes %3f sec' %
                      (self.batch_size, end - start))
                start = end

            # the input and output patch is a 5d numpy array with
            # datatype of float32, the dimensions are batch/channel/z/y/x.
            # the input image should be normalized to [0,1]
            output_patch = self.patch_engine(self.input_patch_buffer)

            if self.verbose:
                assert output_patch.ndim == 5
                end = time.time()
                print('run inference for %d patch takes %3f sec' %
                      (self.batch_size, end - start))
                start = end

            for batch_idx, slices in enumerate(batch_slices):
                # only use the required number of channels
                # the remaining channels are dropped
                output_buffer[(slice(self.num_output_channels), 
                               *slices[1])] += output_patch[batch_idx, :self.num_output_channels, :, :, :]

            if self.verbose:
                end = time.time()
                print('mask and blend patch takes %3f sec' % (end - start))

        if self.verbose:
            print("Inference of whole chunk takes %3f sec" %
                  (time.time() - chunk_time_start))

        #import h5py
        #with h5py.File('/tmp/output_buffer.h5', "w") as f:
        #    f['main'] = output_buffer
        #with h5py.File('/tmp/output_chunk_mask.h5', "w") as f:
        #    f['main'] = self.output_chunk_mask
        #breakpoint()

        if self.mask_output_chunk:
            output_buffer *= self.output_chunk_mask

        # theoretically, all the value of output_buffer should not be greater than 1
        # we use a slightly higher value here to accomondate numerical precision issue
        np.testing.assert_array_less(
            output_buffer,
            1.0001,
            err_msg='output buffer should not be greater than 1')
        return output_buffer

    def _create_output_buffer(self, input_chunk):
        output_buffer = np.zeros(
            (self.num_output_channels, ) + input_chunk.shape, dtype=np.float32)
        return Chunk(output_buffer,
                     global_offset=(0, ) + input_chunk.global_offset)

    def _prepare_patch_engine(self):
        # prepare for inference
        if self.framework == 'pznet':
            from .patch_engine.pznet import PZNet
            self.patch_engine = PZNet(self.convnet_weight_path)
        elif self.framework == 'pytorch':
            from .patch_engine.pytorch import PyTorch
            # currently, we do not support pytorch backend with different
            # input and output patch size and overlap.
            assert self.input_patch_size == self.output_patch_size
            assert self.input_patch_overlap == self.output_patch_overlap
            self.patch_engine = PyTorch(
                self.input_patch_size,
                self.input_patch_overlap,
                self.convnet_model,
                self.convnet_weight_path,
                num_output_channels=self.num_output_channels)
        elif self.framework == 'pytorch-multitask':
            # currently only this type of task support mask in device
            from .patch_engine.pytorch_multitask import PyTorchMultitask
            self.patch_engine = PyTorchMultitask(
                self.convnet_model,
                self.convnet_weight_path,
                input_patch_size=self.input_patch_size,
                output_patch_size=self.output_patch_size,
                output_patch_overlap=self.output_patch_overlap,
                num_output_channels=self.num_output_channels,
                bump=self.bump)
        elif self.framework == 'identity':
            from .patch_engine.identity import Identity
            assert self.input_patch_size == self.output_patch_size
            assert self.input_patch_overlap == self.output_patch_overlap
            self.patch_engine = Identity(
                self.input_patch_size,
                self.output_patch_overlap,
                num_output_channels=self.num_output_channels)
        else:
            raise Exception('invalid inference backend: {}'.format(
                self.framework))
