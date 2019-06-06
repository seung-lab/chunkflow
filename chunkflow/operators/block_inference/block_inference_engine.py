#!/usr/bin/env python
__doc__ = """
Inference a chunk of image
"""

import time
import numpy as np
from tqdm import tqdm 

from .patch_mask import PatchMask
from chunkflow.chunk import Chunk


class BlockInferenceEngine(object):
    """
        BlockInference
    convnet inference for a whole block. the patches should aligned with the \
        block size. 

    """
    def __init__(self, patch_inference_engine, patch_size, patch_overlap,
                 output_key: str='affinity', num_output_channels: int=3, 
                 batch_size: int=1, mask_in_device: bool=False, 
                 verbose: bool=True):
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
        self.verbose = verbose

        # allocate a buffer to avoid redundent 
        self.input_patch_buffer = np.zeros((batch_size, 1, *patch_size), 
                                           dtype=np.float32)
        self.mask_in_device = mask_in_device

    def __call__(self, input_chunk, output_buffer=None):
        """
        args:
            input_chunk (Chunk): input chunk with global offset
        """
        assert isinstance(input_chunk, Chunk)
        if np.all(input_chunk==0):
            print('input is all zero, return zero buffer directly')
            if output_buffer is None:
                output_buffer = self._create_output_buffer(input_chunk)
            return output_buffer 
        
        if input_chunk.dtype == np.uint8:
            input_chunk = input_chunk.astype(np.float32) / 255.0
        
        # patches should be aligned within input chunk
        for i, s, o in zip(input_chunk.shape, self.stride, self.patch_overlap):
            assert (i-o) % s == 0, ('the patche stride {} and overlap {} do not align with the input chunk size {}' % s, o, i)
        
        if self.verbose:
            chunk_time_start = time.time()

        if output_buffer is None:
            # this consumes a lot of memory, should not be preallocated
            output_buffer = self._create_output_buffer(input_chunk)
        
        # contruct input patch offset list
        offset_list = []
        input_size = input_chunk.shape
        input_offset = input_chunk.global_offset
        for oz in tqdm(range(input_offset[0],
                             input_offset[0]+input_size[0]-self.patch_overlap[0],
                             self.stride[0]), 
                       disable=not self.verbose, 
                       desc='ConvNet Inferece: '):
            for oy in range(input_offset[1],
                           input_offset[1]+input_size[1]-self.patch_overlap[1],
                           self.stride[1]):
                for ox in range(input_offset[2],
                                input_offset[2]+input_size[2]-self.patch_overlap[2],
                                self.stride[2]):
                    offset_list.append((oz, oy, ox))
        
        # iterate the offset list
        for i in range(0, len(offset_list), self.batch_size):
            if self.verbose:
                start = time.time()

            offsets = offset_list[i : i + self.batch_size]
            for j,offset in enumerate(offsets):
                input_patch = input_chunk.cutout((
                    slice(offset[0], offset[0] + self.patch_size[0]),
                    slice(offset[1], offset[1] + self.patch_size[1]),
                    slice(offset[2], offset[2] + self.patch_size[2])))
                self.input_patch_buffer[j, :,:,:] = input_patch
            
            if self.verbose:
                end = time.time()
                print('prepare input patch takes %3f sec' % (end - start))
                start = end

            # the input and output patch is a 5d numpy array with
            # datatype of float32, the dimensions are batch/channel/z/y/x.
            # the input image should be normalized to [0,1]
            output_patch = self.patch_inference_engine(self.input_patch_buffer)
            
            if self.verbose:
                end = time.time()
                print('run inference for a patch takes %3f sec' % (end-start))
                start = end

            assert output_patch.ndim == 5 
            for j,offset in enumerate(offsets):
                output_chunk = output_patch[j, :self.num_output_channels, :, :, :]
                
                output_chunk = Chunk(output_chunk, (0,)+offset)

                # normalized by patch mask
                if not self.mask_in_device:
                    output_chunk *= self.patch_mask

                # blend to output buffer
                output_buffer.blend(output_chunk)
            if self.verbose:
                end = time.time()
                print('mask and blend patch takes %3f sec' % (end-start))

        if self.verbose:
            print("Inference of whole chunk takes %3f sec" % 
                  (time.time() - chunk_time_start))

        return output_buffer

    def _create_output_buffer(self, input_chunk):
        assert isinstance(input_chunk, Chunk)
        output_buffer = np.zeros((self.num_output_channels,)+input_chunk.shape,
                                 dtype=np.float32)
        return Chunk(output_buffer,
                     global_offset=(0,)+input_chunk.global_offset)
