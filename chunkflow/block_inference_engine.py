#!/usr/bin/env python
__doc__ = """
Inference a chunk of image
"""

#import time
import numpy as np
from tqdm import tqdm
import asyncio

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
                 output_key='affinity', num_output_channels=3, 
                 is_masked_in_device=False, is_padding=False):
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
        self.is_padding = is_padding 

    def __call__(self, input_chunk, output_buffer=None):
        """
        args:
            input_chunk (OffsetArray): input chunk with global offset
        """
        assert isinstance(input_chunk, OffsetArray)
        
        if np.all(input_chunk==0):
            print('input is all zero, return zero buffer directly')
            if output_buffer is None:
                output_buffer = self._create_output_buffer(input_chunk)
            return output_buffer 
        
        if self.is_padding:
            self._get_pad_size(input_chunk.shape)
            global_offset = input_chunk.global_offset 
            input_chunk = np.pad(input_chunk, ((0, self.pad_size[0]), 
                                               (0, self.pad_size[1]), 
                                               (0, self.pad_size[2])), 'reflect')
            input_chunk = OffsetArray(input_chunk, global_offset)

        if input_chunk.dtype == np.uint8:
            global_offset = input_chunk.global_offset 
            input_chunk = input_chunk.astype(np.float32) / 255.0
            input_chunk = OffsetArray(input_chunk, global_offset)
        
        # patches should be aligned within input chunk
        for i, s, o in zip(input_chunk.shape, self.stride, self.patch_overlap):
            assert (i-o) % s == 0, ('the patche stride {} and overlap {} do not align with the input chunk size {}' % s, o, i)
        
        if output_buffer is None:
            output_buffer = self._create_output_buffer(input_chunk)
        
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
        
        if self.is_padding:
            output_buffer = output_buffer[:, :-self.pad_size[0], 
                                             :-self.pad_size[1],
                                             :-self.pad_size[2]]
        return output_buffer

    def _create_output_buffer(self, input_chunk):
        output_buffer = np.zeros((self.num_output_channels,)+input_chunk.shape,
                                 dtype=np.float32)
        return OffsetArray(output_buffer,
                           global_offset=(0,)+input_chunk.global_offset)

    def _get_pad_size(self, input_chunk_shape):
        """
            _pad_image(self, input_chunk_shape)
        pad the image to make it align with patch size and patch overlap 
        """
        # patches should be aligned within input chunk
        pad_size = [s-(i-o)%s for i, s, o in 
                    zip(input_chunk_shape, self.stride, self.patch_overlap)]
        # if the padding size is the same with stride, then do not need to pad!
        self.pad_size = [ p if s!=p else 0 for s, p in zip(self.stride, pad_size)]
        print('pad size: {}'.format(self.pad_size))


class AsyncBlockInferenceEngine(object):
    """
        BlockInference
    convnet inference for a whole block. the patches should aligned with the \
        block size. Note that async only performs a little bit better than serial 
        (about a few seconds in a few minutes), so it is better to use serial version \
        in case there is any bug here. We can already achieve high utilization using \
        multiple processes.

    Parameters:
        is_masked_in_device: the patch was already masked/normalized in the device, 
                such as gpu, for speed up. 
    """
    def __init__(self,
                 patch_inference_engine,
                 patch_size,
                 patch_overlap,
                 output_key='affinity',
                 num_output_channels=3,
                 is_masked_in_device=False):
        """
        params:
            inference_engine, patch_size, input_chunk, output_key, patch_stride
        """
        assert len(patch_overlap) == 3
        assert len(patch_size) == 3
        self.patch_inference_engine = patch_inference_engine
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.stride = tuple(p - o for p, o in zip(patch_size, patch_overlap))
        self.output_key = output_key
        self.num_output_channels = num_output_channels

        self.patch_mask = PatchMask(patch_size, patch_overlap)
        self.is_masked_in_device = is_masked_in_device

    async def _produce_input_patches(self, input_patch_queue):
        assert isinstance(self.input_chunk, OffsetArray)
        # patches should be aligned within input chunk
        for i, s, o in zip(self.input_chunk.shape, self.stride,
                           self.patch_overlap):
            assert (i - o) % s == 0, (
                'the patche stride {} and overlap {} do not align with the input chunk size {}'.format(
                    s, o, i))

        input_size = self.input_chunk.shape
        input_offset = self.input_chunk.global_offset
        for oz in tqdm(
                range(input_offset[0],
                      input_offset[0] + input_size[0] - self.patch_overlap[0],
                      self.stride[0])):
            for oy in range(
                    input_offset[1],
                    input_offset[1] + input_size[1] - self.patch_overlap[1],
                    self.stride[1]):
                for ox in range(
                        input_offset[2], input_offset[2] + input_size[2] -
                        self.patch_overlap[2], self.stride[2]):
                    input_patch = self.input_chunk.cutout(
                        (slice(oz, oz + self.patch_size[0]),
                         slice(oy, oy + self.patch_size[1]),
                         slice(ox, ox + self.patch_size[2])))
                    await input_patch_queue.put(input_patch)
        # notify the worker to finish processing
        await input_patch_queue.put(None)
        print('finish producing input patches!')

    async def _inference_worker(self, input_patch_queue, output_patch_queue):
        # the input and output patch is a 5d numpy array with
        # datatype of float32, the dimensions are batch/channel/z/y/x.
        # the input image should be normalized to [0,1]
        while True:
            input_patch = await input_patch_queue.get()
            if input_patch is None:
                await output_patch_queue.put(None)
                input_patch_queue.task_done()
                break

            output_patch = self.patch_inference_engine(input_patch)

            # remove the batch number dimension
            output_patch = np.squeeze(output_patch, axis=0)

            output_patch = output_patch[:self.num_output_channels, :, :, :]

            output_patch = OffsetArray(output_patch,
                                       (0, ) + input_patch.global_offset)

            # normalized by patch mask
            if not self.is_masked_in_device:
                output_patch *= self.patch_mask
            await output_patch_queue.put(output_patch)
            input_patch_queue.task_done()

        print('finish patch inference!')

    async def _blend_output_patches(self, output_patch_queue):
        while True:
            output_patch = await output_patch_queue.get()
            if output_patch is None:
                # get the nofication that this is the end
                output_patch_queue.task_done()
                break

            # blend to output buffer
            self.output_buffer.blend(output_patch)
            output_patch_queue.task_done()
        print('finish blending output patches!')

    def __call__(self, input_chunk, output_buffer=None):
        """
        args:
            input_chunk (OffsetArray): input chunk with global offset
        """
        self.input_chunk = input_chunk
        self.output_buffer = output_buffer

        if self.output_buffer is None:
            self.output_buffer = self._create_output_buffer(self.input_chunk)
        if np.all(self.input_chunk == 0):
            print('input is all zero, return zero buffer directly')
            return self.output_buffer

        if self.input_chunk.dtype == np.uint8:
            global_offset = self.input_chunk.global_offset
            self.input_chunk = np.asarray(
                self.input_chunk, dtype='float32') / 255.0
            self.input_chunk = OffsetArray(self.input_chunk, global_offset)

        input_patch_queue = asyncio.Queue(maxsize=1)
        output_patch_queue = asyncio.Queue(maxsize=1)

        loop = asyncio.get_event_loop()
        task1 = loop.create_task(
            self._produce_input_patches(input_patch_queue))
        task2 = loop.create_task(
            self._inference_worker(input_patch_queue, output_patch_queue))
        task3 = loop.create_task(
            self._blend_output_patches(output_patch_queue))

        loop.run_until_complete(task1)
        loop.run_until_complete(task2)
        loop.run_until_complete(task3)
        #loop.close()

        return self.output_buffer

    def _create_output_buffer(self, input_chunk):
        output_buffer = np.zeros(
            (self.num_output_channels, ) + input_chunk.shape, dtype=np.float32)
        return OffsetArray(
            output_buffer, global_offset=(0, ) + input_chunk.global_offset)
