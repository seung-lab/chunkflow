#!/usr/bin/env python
__doc__ = """
Inference a chunk of image
"""

from frameworks.pytorch import PyTorchEngine
import time
import numpy as np

from dataprovider.offset_array import OffsetArray
from dataprovider.dataset import VolumeDataset, TensorData
from dataprovider.inference.forward import AlignedPatchForwardScanner


class BlockInferenceEngine(object):
    """
        BlockInference
    convnet inference for a whole block. the patches should aligned with the \
        block size.
    """
    def __init__(self, patch_inference_engine, patch_size, output_key,
                 output_channels, patch_stride_percentile):
        """
        params:
            inference_engine, patch_size, input_chunk, output_key, patch_stride
        """
        self.patch_inference_engine = patch_inference_engine
        self.patch_size = patch_size
        self.scan_spec = {output_key: (output_channels,)+self.patch_size}
        # initialize dataset
        self.dataset = VolumeDataset()

        self.patch_stride_percentile = patch_stride_percentile
        assert all(np.less_equal(self.patch_stride_percentile, 1))

        self.scan_params = dict(stride=self.patch_stride_percentile,
                           blend='aligned-bump')


    def __call__(self, input_chunk, input_key='input'):
        self._set_input_chunk(input_key, input_chunk)
        self.scanner = AlignedPatchForwardScanner(self.dataset, self.scan_spec,
                                      params=self.scan_params)

        start = time.time()
        #for oz, oy, ox in zip(range(0,self.stride))
        while True:
            inputs = self.scanner.pull()
            if inputs is None:
                break

            # the output is a 5d/3d numpy array with datatype of float32
            # the image should be normalized to [0,1]
            output = self.patch_inference_engine(list(inputs.values())[0])

            # push output patch to chunk
            fmt_outputs = dict()
            for (i, k) in enumerate(self.scan_spec.keys()):
                fmt_outputs[k] = output[0, :3, :, :, :]
                # setup the location output
                # assume that the ZYX shape of input and output is the same
                assert output.shape[-3:] == list(inputs.values())[0].shape[-3:]
                loc = list(inputs.values())[0].global_offset
                fmt_outputs[k] = OffsetArray(fmt_outputs[k], global_offset=loc)

            self.scanner.push(fmt_outputs)

            end = time.time()
            print("Elapsed: %3f sec" % (end-start))
            start = end
        return self.scanner.outputs.data

    def _set_input_chunk(self, input_key, input_chunk):
        """
        Params:
            key: string, the key name of input
            input_chunk: 3D numpy array
        """
        tensor = TensorData(input_chunk, fov=self.patch_size)
        self.dataset.add_data(input_key, tensor)
        self.dataset.set_spec(dict(input=self.patch_size))


if __name__ == '__main__':
    model_file_name = '/usr/people/jingpeng/seungmount/research/kisuklee/Workbench/torms3/pinky-pytorch/code/rsunet.py'
    net_file_name = './frameworks/model200000.chkpt'
    engine = PyTorchEngine(model_file_name, net_file_name)

    from dataprovider.emio import imsave
    import h5py
    fimg = '/usr/people/jingpeng/seungmount/research/kisuklee/Workbench/deep_learning/kaffe/datasets/pinky/ground_truth/stitched/img.h5'
    with h5py.File(fimg, 'r') as f:

        img = f['main'][:18+15*2, :256+205*2, :256+205*2]
        imsave(img, '/tmp/img.h5')
        img = np.asarray(img, dtype='float32') / 255.0
        inference = BlockInferenceEngine(
            patch_inference_engine=engine,
            patch_size=(18, 256, 256),
            output_key='affinity',
            output_channels = 3,
            patch_stride_percentile=(0.8, 0.8, 0.8))

        output_sample = inference(img)
        for k, tensor in output_sample.items():
            print('shape of output: {}'.format(tensor.data.shape))
            imsave(tensor.data[0, :, :, :], '/tmp/{}.tif'.format(k))
