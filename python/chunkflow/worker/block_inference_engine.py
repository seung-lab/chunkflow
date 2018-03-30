#!/usr/bin/env python
__doc__ = """
Inference a chunk of image
"""

from frameworks.pytorch import PyTorchEngine
import time
import numpy as np

from chunkflow.worker.patch_mask import PatchMask
#from .patch_mask import PatchMask
from chunkflow.worker.offset_array import OffsetArray


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
        self.patch_inference_engine = patch_inference_engine
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = tuple(p-o for p, o in zip(patch_size, overlap))
        self.output_key = output_key
        self.output_channels = output_channels

        self.patch_mask = PatchMask(patch_size, overlap)

    def __call__(self, input_chunk):
        """
        args:
            input_chunk (OffsetArray): input chunk with global offset
        """
        assert isinstance(input_chunk, OffsetArray)
        # patches should be aligned within input chunk
        for i, s, o in zip(input_chunk.shape, self.stride, self.overlap):
            assert i%s == o
        output_buffer = np.zeros((self.output_channels,)+input_chunk.shape,
                                 dtype=input_chunk.dtype)
        output_buffer = OffsetArray(output_buffer,
                                    (0,)+input_chunk.global_offset)

        start = time.time()
        input_size = input_chunk.shape
        input_offset = input_chunk.global_offset
        for oz in range(input_offset[0],
                        input_offset[0]+input_size[0]-self.overlap[0],
                        self.stride[0]):
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

                    # the output is a 4d numpy array with datatype of float32
                    # the input image should be normalized to [0,1]
                    print('shape of input: {}'.format(input_patch.shape))
                    output_patch = self.patch_inference_engine(input_patch)
                    output_patch = output_patch[:self.output_channels, :, :, :]
                    output_patch = OffsetArray(output_patch,
                                               (0,)+input_patch.global_offset)

                    # normalized by patch mask
                    output_patch *= self.patch_mask

                    # blend to output buffer
                    output_buffer.blend(output_patch)
                    end = time.time()
                    print("Elapsed: %3f sec" % (end-start))
                    start = end
        return output_buffer


if __name__ == '__main__':
    model_file_name = '/usr/people/jingpeng/seungmount/research/kisuklee/Workbench/torms3/pinky-pytorch/code/rsunet.py'
    net_file_name = './frameworks/model200000.chkpt'
    engine = PyTorchEngine(model_file_name, net_file_name)

    from dataprovider.emio import imsave
    import h5py
    #fimg = '/usr/people/jingpeng/seungmount/research/kisuklee/Workbench/deep_learning/kaffe/datasets/pinky/ground_truth/stitched/img.h5'
    fimg = '/tmp/img.h5'
    with h5py.File(fimg, 'r') as f:

        img = f['main'][:18+14*1, :256+204*1, :256+204*1]
        imsave(img, '/tmp/img.tif')
        img = np.asarray(img, dtype='float32') / 255.0
        img = OffsetArray(img)
        inference = BlockInferenceEngine(
            patch_inference_engine=engine,
            patch_size=(18, 256, 256),
            overlap = (4, 52, 52),
            output_key='affinity',
            output_channels = 3)

        output = inference(img)
        print('shape of output: {}'.format(output.shape))
        imsave(output[0, :, :, :], '/tmp/out.tif')
