#!/usr/bin/env python
__doc__ = """
Inference a chunk of image
"""

import time
import numpy as np

from dataprovider.dataset import VolumeDataset, TensorData
from dataprovider import ForwardScanner


class AlignedPatchInference(object):
    """
        AlignedPatchInference

    """
    def __init__(self, **kwargs):
        """
        params:
            inference_engine, patch_size, input_chunk, output_key, patch_stride
        """
        self.inference_engine = kwargs['inference_engine']
        self.patch_size = kwargs['patch_size']
        self.scan_spec = eval('dict({}={})'.format(kwargs['output_key'],
                                                   (3,)+self.patch_size))
        # initialize dataset
        self.dataset = VolumeDataset()
        self.dataset.add_raw_data(key='input', data=kwargs['input_chunk'])
        self.dataset.set_spec(dict(input=self.patch_size))

        self.patch_stride_percentile = kwargs['patch_stride_percentile']
        assert all(np.less_equal(self.patch_stride_percentile, 1))

        scan_params = dict(stride=self.patch_stride_percentile,
                           blend='aligned-bump')

        self.scanner = ForwardScanner(self.dataset, self.scan_spec,
                                      params=scan_params)

    def __call__(self, input_key, input_chunk):
        self._set_input_chunk(input_key, input_chunk)

        start = time.time()
        while True:
            inputs = self.scanner.pull()
            if inputs is None:
                break

            # the output is a 5d/3d numpy array with datatype of float32
            # the image should be normalized to [0,1]
            output = self.inference_engine(list(inputs.values())[0])
            assert np.all(np.less_equal(output, 1.0))

            # push output patch to chunk
            fmt_outputs = dict()
            for (i, k) in enumerate(self.scan_spec.keys()):
                fmt_outputs[k] = output[0, :3, :, :, :]

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


if __name__ == '__main__':
    from frameworks.pytorch import PyTorchEngine
    model_file_name = '/usr/people/jingpeng/seungmount/research/kisuklee/\
        Workbench/torms3/pinky-pytorch/code/rsunet.py'
    net_file_name = './frameworks/model200000.chkpt'
    engine = PyTorchEngine(model_file_name, net_file_name)

    from dataprovider.emio import imsave
    import h5py
    fimg = '/usr/people/jingpeng/seungmount/research/kisuklee/Workbench/\
        deep_learning/kaffe/datasets/pinky/ground_truth/stitched/img.h5'
    with h5py.File(fimg, 'r') as f:
        img = f['main'][:18+15*1, :256+205*1, :256+205*1]
        imsave(img, '/tmp/img.h5')
        img = np.asarray(img, dtype='float32') / 255.0
        inference = AlignedPatchInference(
            inference_engine=engine,
            patch_size=(18, 256, 256),
            output_key='affinity',
            input_chunk=img,
            patch_stride_percentile=(0.8, 0.8, 0.8))

        output_sample = inference('input', img)
        for k, tensor in output_sample.items():
            print('shape of output: {}'.format(tensor.data.shape))
            imsave(tensor.data[0, :, :, :], '/tmp/{}.tif'.format(k))
