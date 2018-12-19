#!/usr/bin/env python
__doc__ = """
Inference a chunk of image
"""

#import time
import numpy as np
from tqdm import tqdm

from .patch_mask import PatchMask
from .offset_array import OffsetArray


class BlockInferenceEngine(object):
    """
        BlockInference
    convnet inference for a whole block. the patches should aligned with the \
        block size. 

    Parameters:
        is_masked_in_device: the patch was already masked/normalized in the device, 
                such as gpu, for speed up. 
    """
    def __init__(self, patch_inference_engine, patch_size, patch_overlap,
                 output_key='affinity', num_output_channels=3, is_masked_in_device=False):
        """
        params:
            inference_engine, patch_size, input_chunk, output_key, patch_stride
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
        self.is_masked_in_device = is_masked_in_device

    def __call__(self, input_chunk, output_buffer=None):
        """
        args:
            input_chunk (OffsetArray): input chunk with global offset
        """
        if output_buffer is None:
            output_buffer = self._create_output_buffer(input_chunk)
    
        if np.all(input_chunk==0):
            print('input is all zero, return zero buffer directly')
            return output_buffer 

        if input_chunk.dtype == 'uint8':
            global_offset = input_chunk.global_offset 
            input_chunk = np.asarray(input_chunk, dtype='float32') / 255.0
            input_chunk = OffsetArray(input_chunk, global_offset)

        assert isinstance(input_chunk, OffsetArray)
        # patches should be aligned within input chunk
        for i, s, o in zip(input_chunk.shape, self.stride, self.patch_overlap):
            assert (i-o) % s == 0, ('the patche stride {} and overlap {} do not align with the input chunk size {}' % s, o, i)

        #start = time.time()
        input_size = input_chunk.shape
        input_offset = input_chunk.global_offset
        for oz in tqdm(range(input_offset[0],
                             input_offset[0]+input_size[0]-self.patch_overlap[0],
                             self.stride[0])):
            for oy in range(input_offset[1],
                           input_offset[1]+input_size[1]-self.patch_overlap[1],
                           self.stride[1]):
                for ox in range(input_offset[2],
                                input_offset[2]+input_size[2]-self.patch_overlap[2],
                                self.stride[2]):
                    input_patch = input_chunk.cutout((
                        slice(oz, oz + self.patch_size[0]),
                        slice(oy, oy + self.patch_size[1]),
                        slice(ox, ox + self.patch_size[2])))

                    # the input and output patch is a 5d numpy array with
                    # datatype of float32, the dimensions are batch/channel/z/y/x.
                    # the input image should be normalized to [0,1]
                    output_patch = self.patch_inference_engine(input_patch)

                    # remove the batch number dimension
                    output_patch = np.squeeze(output_patch, axis=0)

                    output_patch = output_patch[:self.num_output_channels, :, :, :]
                    
                    output_patch = OffsetArray(output_patch,
                                               (0,)+input_patch.global_offset)

                    # normalized by patch mask
                    if not self.is_masked_in_device:
                        output_patch *= self.patch_mask

                    # blend to output buffer
                    output_buffer.blend(output_patch)
                    #end = time.time()
                    #print("Elapsed: %3f sec" % (end-start))
                    #start = end
        return output_buffer

    def _create_output_buffer(self, input_chunk):
        output_buffer = np.zeros((self.num_output_channels,)+input_chunk.shape,
                                 dtype=np.float32)
        return OffsetArray(output_buffer,
                           global_offset=(0,)+input_chunk.global_offset)

