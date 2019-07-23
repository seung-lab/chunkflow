#!/usr/bin/env python
__doc__ = """
Inference a chunk of image
"""

import time
import numpy as np
from tqdm import tqdm
from warnings import warn

from .patch_mask import PatchMask
from chunkflow.chunk import Chunk


class BlockInferenceEngine(object):
    """
        BlockInference
    convnet inference for a whole block. the patches should aligned with the \
        block size.

    """
    def __init__(self, patch_inference_engine, patch_size: tuple,
                 patch_overlap: tuple, output_key: str = 'affinity',
                 num_output_channels: int = 3, batch_size: int = 1,
                 input_size: tuple = None, mask_output_chunk: bool = False,
                 mask_in_device: bool = False, verbose: bool = True):
        """
        params:
            patch_inference_engine: inference for each patch.
            patch_size: (tuple) the size of patch.
            input_chunk: (ndarray) input image
            output_key: (str) the key of ConvNet output
            patch_overlap: (tuple of int) overlap of each patch
            num_output_channels: (int) the number of output channels.
            batch_size: (int) the mini batch size 
            verbose: (bool) print out info or not.
        """
        assert len(patch_overlap) == 3
        assert len(patch_size) == 3
        self.patch_inference_engine = patch_inference_engine
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.stride = tuple(p-o for p, o in zip(patch_size, patch_overlap))
        self.output_key = output_key
        self.num_output_channels = num_output_channels

        self.patch_mask = PatchMask(patch_size, patch_overlap)
        self.batch_size = batch_size
        self.input_size = input_size
        self.verbose = verbose

        self.mask_in_device = mask_in_device
        self.mask_output_chunk = mask_output_chunk

        # allocate a buffer to avoid redundent memory allocation 
        self.input_patch_buffer = np.zeros((batch_size, 1, *patch_size), 
                                           dtype=np.float32)

    def _check_alignment(self):
        if not self.mask_output_chunk:
            if self.verbose:
                print('patches should align with input chunk when not using output chunk mask.')
            is_align = tuple((i-o)%s==0 for i, s, o in 
                             zip(self.input_size, self.stride, 
                                 self.patch_overlap))
            # all axis should be aligned
            # the patches should aligned with input size in case 
            # we will not mask the output chunk
            assert np.all(is_align)

    def _construct_patch_slice_list(self):
        """
        create the normalization mask and patch bounding box list
        """
        self.patch_slice_list = []
        # the step is the stride, so the end of aligned patch is input_size - patch_overlap
        for oz in tqdm(range(0, self.input_size[0]-self.patch_overlap[0], self.stride[0]), 
                       disable=not self.verbose, 
                       desc='ConvNet Inferece: '):
            if oz + self.patch_size[0] > self.input_size[0]:
                oz = self.input_size[0] - self.patch_size[0]
                assert oz >= 0
            for oy in range(0, self.input_size[1]-self.patch_overlap[1], self.stride[1]):
                if oy + self.patch_size[1] > self.input_size[1]:
                    oy = self.input_size[1] - self.patch_size[1]
                    assert oy >= 0
                for ox in range(0, self.input_size[2]-self.patch_overlap[2], self.stride[2]):
                    if ox + self.patch_size[2] > self.input_size[2]:
                        ox = self.input_size[2] - self.patch_size[2]
                        assert ox >= 0
                    patch_slice = (slice(oz, oz+self.patch_size[0]),
                                   slice(oy, oy+self.patch_size[1]),
                                   slice(ox, ox+self.patch_size[2]))
                    self.patch_slice_list.append(patch_slice)

    def _construct_output_chunk_mask(self):
        if not self.mask_output_chunk:
            return

        if self.verbose:
            print('creating output chunk mask...')

        self.output_chunk_mask = np.zeros(self.input_size[-3:], np.float32)
        for patch_slices in self.patch_slice_list:
            # accumulate weights
            self.output_chunk_mask[patch_slices] += self.patch_mask
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
            assert self.patch_slice_list is not None 
            assert self.output_chunk_mask is not None 
        else:
            if self.input_size is not None:
                warn('the input size has changed, using new intput size.')
            self.input_size = input_size
            self._construct_patch_slice_list()
            if self.mask_output_chunk:
                self._construct_output_chunk_mask()

    def __call__(self, input_chunk: np.ndarray, output_buffer: np.ndarray=None):
        """
        args:
            input_chunk (Chunk): input chunk with global offset
        """
        assert isinstance(input_chunk, Chunk)
           
        if output_buffer is None:
            output_buffer = self._create_output_buffer(input_chunk)

        if np.all(input_chunk==0):
            print('input is all zero, return zero buffer directly')
            return output_buffer
        
        self._check_input_size_and_prepare_data(input_chunk.shape)
        self._check_alignment()
         
        if input_chunk.dtype == np.uint8:
            input_chunk = input_chunk.astype(np.float32) / 255.0
        
        if self.verbose:
            chunk_time_start = time.time()
               
        # iterate the offset list
        for i in tqdm(range(0, len(self.patch_slice_list), self.batch_size),
                      disable=not self.verbose,
                      desc='ConvNet Inference ...'):
            if self.verbose:
                start = time.time()

            patch_slices = self.patch_slice_list[i:i + self.batch_size]
            for j,slices in enumerate(patch_slices):
                self.input_patch_buffer[j, 0, :, :, :] = input_chunk[slices]
            
            if self.verbose:
                end = time.time()
                print('prepare %d input patches takes %3f sec' % (self.batch_size, end - start))
                start = end

            # the input and output patch is a 5d numpy array with
            # datatype of float32, the dimensions are batch/channel/z/y/x.
            # the input image should be normalized to [0,1]
            output_patch = self.patch_inference_engine(self.input_patch_buffer)

            if self.verbose:
                assert output_patch.ndim == 5
                end = time.time()
                print('run inference for %d patch takes %3f sec' % 
                      (self.batch_size, end-start))
                start = end

            for j,slices in enumerate(patch_slices):
                # only use the required number of channels
                # the remaining channels are dropped
                output_chunk = output_patch[j, :self.num_output_channels,
                                            :, :, :]

                # normalized by patch mask
                if not self.mask_in_device:
                    output_chunk *= self.patch_mask
                
                output_buffer[((slice(self.num_output_channels)), 
                               *slices)] += output_chunk

            if self.verbose:
                end = time.time()
                print('mask and blend patch takes %3f sec' % (end-start))

        if self.verbose:
            print("Inference of whole chunk takes %3f sec" % 
                  (time.time() - chunk_time_start))
        
        if self.mask_output_chunk:
            output_buffer *= self.output_chunk_mask
    
        # theoretically, all the value of output_buffer should not be greater than 1
        # we use a slightly higher value here to accomondate numerical precision issue
        np.testing.assert_array_less(output_buffer, 1.0001, 
                                     err_msg='output buffer should not be greater than 1')
        return output_buffer

    def _create_output_buffer(self, input_chunk):
        output_buffer = np.zeros((self.num_output_channels,)+input_chunk.shape,
                                 dtype=np.float32)
        return Chunk(output_buffer,
                     global_offset=(0,)+input_chunk.global_offset)


