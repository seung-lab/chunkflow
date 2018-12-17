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
    """
    def __init__(self, patch_inference_engine, patch_size, overlap,
                 output_key='affinity', output_channels=3):
        """
        params:
            inference_engine, patch_size, input_chunk, output_key, patch_stride
        """
        assert len(overlap) == 3
        assert len(patch_size) == 3
        self.patch_inference_engine = patch_inference_engine
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = tuple(p-o for p, o in zip(patch_size, overlap))
        self.output_key = output_key
        self.output_channels = output_channels

        self.patch_mask = PatchMask(patch_size, overlap)

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
        for i, s, o in zip(input_chunk.shape, self.stride, self.overlap):
            assert (i-o) % s == 0

        #start = time.time()
        input_size = input_chunk.shape
        input_offset = input_chunk.global_offset
        for oz in tqdm(range(input_offset[0],
                             input_offset[0]+input_size[0]-self.overlap[0],
                             self.stride[0])):
            for oy in range(input_offset[1],
                           input_offset[1]+input_size[1]-self.overlap[1],
                           self.stride[1]):
                for ox in range(input_offset[2],
                                input_offset[2]+input_size[2]-self.overlap[2],
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

                    output_patch = output_patch[:self.output_channels, :, :, :]
                    
                    output_patch = OffsetArray(output_patch,
                                               (0,)+input_patch.global_offset)

                    # normalized by patch mask
                    output_patch *= self.patch_mask

                    # blend to output buffer
                    output_buffer.blend(output_patch)
                    #end = time.time()
                    #print("Elapsed: %3f sec" % (end-start))
                    #start = end
        return output_buffer

    def _create_output_buffer(self, input_chunk):
        output_buffer = np.zeros((self.output_channels,)+input_chunk.shape,
                                 dtype=np.float32)
        return OffsetArray(output_buffer,
                           global_offset=(0,)+input_chunk.global_offset)


if __name__ == '__main__':
    from frameworks.pytorch import PyTorchEngine
    model_file_name = '/usr/people/jingpeng/seungmount/research/kisuklee/\
        Workbench/torms3/pinky-pytorch/code/rsunet.py'
    net_file_name = './frameworks/model200000.chkpt'
    engine = PyTorchEngine(model_file_name, net_file_name)

    from emirt.emio import imsave
    import h5py
    fimg = '/tmp/img.h5'
    with h5py.File(fimg, 'r') as f:
        img = f['main'][:32+28*0, :256+192*1, :256+192*1]
        imsave(img, '/tmp/img.tif')
        img = np.asarray(img, dtype='float32') / 255.0
        img = OffsetArray(img)
        inference = BlockInferenceEngine(
            patch_inference_engine=engine,
            patch_size=(32, 256, 256),
            overlap=(4, 64, 64),
            output_key='affinity',
            output_channels=3)

        output = inference(img)
        print('shape of output: {}'.format(output.shape))
        imsave(output[0, :, :, :], '/tmp/out.tif')
