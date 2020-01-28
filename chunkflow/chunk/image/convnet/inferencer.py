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


class Inferencer(object):
    """
        Inferencer
    convnet inference for a whole chunk. the patches should aligned with
    the output chunk size.
    """
    def __init__(self,
                 convnet_model: str,
                 convnet_weight_path: str,
                 input_patch_size: Union[tuple, list],
                 output_patch_size: Union[tuple, list] = None,
                 num_output_channels: int = 3,
                 output_patch_overlap: Union[tuple, list] = (4, 64, 64),
                 output_chunk_start_offset: tuple = (0, 0, 0),
                 framework: str = 'identity',
                 batch_size: int = 1,
                 bump: str = 'wu',
                 input_size: tuple = None,
                 mask_output_chunk: bool = False,
                 verbose: bool = False):

        self.batch_size = batch_size
        self.input_size = input_size
        self.verbose = verbose
        self.mask_output_chunk = mask_output_chunk
        self.output_chunk_start_offset = output_chunk_start_offset
        
        # allocate a buffer to avoid redundant memory allocation
        self.input_patch_buffer = np.zeros((batch_size, 1, *input_patch_size),
                                           dtype=np.float32)

        self.patch_slices_list = []
        self.output_buffer = None

        self._prepare_patch_inferencer(framework, convnet_model, convnet_weight_path,
                                   input_patch_size, output_patch_size,
                                   output_patch_overlap, num_output_channels,
                                   bump)
    
        if framework in ('pznet', 'identity'):
            import platform
            self.compute_device = platform.processor()
        else:
            import torch
            self.compute_device = torch.cuda.get_device_name(0)
    
    def _prepare_patch_inferencer(self, framework, convnet_model, convnet_weight_path, 
                              input_patch_size, output_patch_size, 
                              output_patch_overlap, num_output_channels, bump):
        # prepare for inference
        if framework == 'pznet':
            from .patch.pznet import PZNet as PatchInferencer
        elif framework == 'pytorch':
            from .patch.pytorch import PyTorch as PatchInferencer
            # currently, we do not support pytorch backend with different
            # input and output patch size and overlap.
        elif framework == 'pytorch-multitask':
            # currently only this type of task support mask in device
            from .patch.pytorch_multitask import PyTorchMultitask as PatchInferencer
        elif framework == 'identity':
            from .patch.identity import Identity as PatchInferencer
        else:
            raise Exception(f'invalid inference backend: {self.framework}')
        
        self.patch_inferencer = PatchInferencer(
            convnet_model,
            convnet_weight_path,
            input_patch_size=input_patch_size,
            output_patch_size=output_patch_size,
            output_patch_overlap=output_patch_overlap,
            num_output_channels=num_output_channels,
            bump=bump)

    def _check_alignment(self):
        if self.verbose:
            print('patches should align with input chunk when ' +
                  'not using output chunk mask.')
        is_align = tuple((i - o) % s == 0 for i, s, o in zip(
            self.input_size, 
            self.patch_inferencer.input_patch_stride, 
            self.patch_inferencer.input_patch_overlap))

        # all axis should be aligned
        # the patches should aligned with input size in case
        # we will not mask the output chunk
        assert np.all(is_align)

    def _construct_patch_slices_list(self, input_chunk_offset):
        """
        create the normalization mask and patch bounding box list
        """
        self.patch_slices_list = []
        # the step is the stride, so the end of aligned patch is
        # input_size - patch_overlap
        
        input_patch_size = self.patch_inferencer.input_patch_size
        output_patch_size = self.patch_inferencer.output_patch_size
        input_patch_overlap = self.patch_inferencer.input_patch_overlap 
        input_patch_stride = self.patch_inferencer.input_patch_stride 
        output_patch_crop_margin_size = self.patch_inferencer.output_patch_crop_margin_size

        for iz in tqdm(range(0, self.input_size[0] - input_patch_overlap[0], input_patch_stride[0]),
                       disable=not self.verbose, desc='ConvNet Inferece: '):
            if iz + input_patch_size[0] > self.input_size[0]:
                iz = self.input_size[0] - input_patch_size[0]
                assert iz >= 0
            iz += input_chunk_offset[-3]
            oz = iz + output_patch_crop_margin_size[0]
            for iy in range(0, self.input_size[1] - input_patch_overlap[1], input_patch_stride[1]):
                if iy + input_patch_size[1] > self.input_size[1]:
                    iy = self.input_size[1] - input_patch_size[1]
                    assert iy >= 0
                iy += input_chunk_offset[-2]
                oy = iy + output_patch_crop_margin_size[1]
                for ix in range(0, self.input_size[2] - input_patch_overlap[2], input_patch_stride[2]):
                    if ix + input_patch_size[2] > self.input_size[2]:
                        ix = self.input_size[2] - input_patch_size[2]
                        assert ix >= 0
                    ix += input_chunk_offset[-1]
                    ox = ix + output_patch_crop_margin_size[2]
                    input_patch_slice =  (slice(iz, iz + input_patch_size[0]),
                                          slice(iy, iy + input_patch_size[1]),
                                          slice(ix, ix + input_patch_size[2]))
                    output_patch_slice = (slice(oz, oz + output_patch_size[0]),
                                          slice(oy, oy + output_patch_size[1]),
                                          slice(ox, ox + output_patch_size[2]))
                    self.patch_slices_list.append((input_patch_slice, output_patch_slice))

    def _construct_output_chunk_mask(self):
        if not self.mask_output_chunk:
            return

        if self.verbose:
            print('creating output chunk mask...')

        self.output_chunk_mask = np.zeros(self.output_size[-3:], np.float32)
        assert len(self.patch_slices_list) > 0
        for _, output_patch_slice in self.patch_slices_list:
            # accumulate weights using the patch mask in RAM
            self.output_chunk_mask[output_patch_slice] += self.patch_inferencer.output_patch_mask_numpy

        # normalize weight, so accumulated inference result multiplies

        # this mask will result in 1
        self.output_chunk_mask = 1.0 / self.output_chunk_mask
    
    def _update_output_buffer(self, input_chunk):

        if self.output_buffer is None:
            self.output_buffer = np.zeros(
                (self.patch_inferencer.num_output_channels, ) +
                self.output_size, dtype=np.float32)
        else:
            # we have to make sure that all the value is 0 initially
            # so we can add patches on it.
            self.output_buffer.fill(0)

        output_offset = tuple(
            io + ocso for io, ocso in
            zip(input_chunk.global_offset,
                self.output_chunk_start_offset))

        self.output_buffer = Chunk(self.output_buffer,
                                   global_offset=(0,) + output_offset)
        return

    def _check_input_size_and_prepare_data(self, input_chunk):
        """
        if the input size is consistent with old one, reuse the
        patch offset list and output chunk mask. Otherwise, recompute them.
        """
        input_size = input_chunk.shape
        if self.input_size == input_size:
            print('reusing output chunk mask.')
            assert self.patch_slices_list is not None
        else:
            if self.input_size is not None:
                warn('the input size has changed, using new intput size.')
            self.input_size = input_size
            
            self.output_size = tuple(
                isz-2*ocso for isz, ocso in 
                zip(input_size, self.output_chunk_start_offset))
        
        self._construct_patch_slices_list(input_chunk.global_offset)
        self._construct_output_chunk_mask()
        self._update_output_buffer(input_chunk)


    def __call__(self, input_chunk: np.ndarray):
        """
        args:
            input_chunk (Chunk): input chunk with global offset
        """
        assert isinstance(input_chunk, Chunk)
        
        self._check_input_size_and_prepare_data(input_chunk)
        if not self.mask_output_chunk:
            self._check_alignment()

        if np.all(input_chunk == 0):
            print('input is all zero, return zero buffer directly')
            return self.output_buffer

        if np.issubdtype(input_chunk.dtype, np.integer):
            # normalize to 0-1 value range
            dtype_max = np.iinfo(input_chunk.dtype).max
            input_chunk = input_chunk.astype(np.float32) / dtype_max

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
                self.input_patch_buffer[
                    batch_idx, 0, :, :, :] = input_chunk.cutout(slices[0])

            if self.verbose:
                end = time.time()
                print('prepare %d input patches takes %3f sec' %
                      (self.batch_size, end - start))
                start = end

            # the input and output patch is a 5d numpy array with
            # datatype of float32, the dimensions are batch/channel/z/y/x.
            # the input image should be normalized to [0,1]
            output_patch = self.patch_inferencer(self.input_patch_buffer)

            if self.verbose:
                assert output_patch.ndim == 5
                end = time.time()
                print('run inference for %d patch takes %3f sec' %
                      (self.batch_size, end - start))
                start = end

            for batch_idx, slices in enumerate(batch_slices):
                # only use the required number of channels
                # the remaining channels are dropped
                offset = (0,) + tuple(s.start for s in slices[1])
                output_chunk = Chunk(output_patch[batch_idx, :, :, :, :],
                                     global_offset=offset)
                self.output_buffer.blend(output_chunk)

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

        if self.mask_output_chunk:
            self.output_buffer *= self.output_chunk_mask

        # theoretically, all the value of output_buffer should not be greater than 1
        # we use a slightly higher value here to accomondate numerical precision issue
        np.testing.assert_array_less(self.output_buffer, 1.0001,
            err_msg='output buffer should not be greater than 1')

        return self.output_buffer
