#!/usr/bin/env python
import torch
import numpy as np
import os
from offset_array import OffsetArray
from cloudvolume.cloudvolume import CloudVolume
from cloudvolume.storage import Storage
from block_inference_engine import BlockInferenceEngine
from frameworks.patch_inference_engine import PatchInferenceEngine
from chunk_manager import ChunkManager
from options import InferenceAndDonateOptions


class InferenceAndDonate(object):
    """
        InferenceTask
    convnet inference for a chunk. This is the first pass of inference, which
    will donate the margin chunks to exchange storage.
    args:
        input_volume (CloudVolume or OffsetArray): could be used to cutout or \
            save sparsly in global coordinate system
        output_volume: similar with input_volume
        exchange_storage (key-value store or dict): exchange the shared chunks.
        output_block_slices (tuple of slices): have to be aligned with output \
            CloudVolume block size
        overlap (tuple of int): the margin size blending with neighboring \
            patches/blocks.
        inference_engine (PatchInferenceEngine, BlockInferenceEngine): \
            callable class to transform image to convnet output, \
            such as affinitymap or psd map.
    """
    def __init__(self, input_volume, output_volume, exchange_storage,
                 output_block_slices, overlap, inference_engine,
                 output_channels=3):
        self.input_volume = input_volume
        self.output_volume = output_volume
        self.exchange_storage = exchange_storage
        self.output_block_slices = output_block_slices
        self.overlap = overlap
        self.inference_engine = inference_engine

        output_buffer_size = (output_channels,) + tuple(
            s.stop-s.start+o for s, o in zip(output_block_slices, overlap))
        output_buffer_offset = (0, ) + tuple(
            s.start for s in output_block_slices)
        self.output_buffer = OffsetArray(np.zeros(output_buffer_size),
                                         global_offset=output_buffer_offset)

        self.chunk_manager = ChunkManager(self.output_buffer,
                                          self.output_volume, exchange_storage,
                                          output_block_slices, overlap)
        self._check_params()

    def _check_params(self):
        assert isinstance(self.input_volume, CloudVolume) or \
            isinstance(self.input_volume, OffsetArray)
        assert isinstance(self.output_volume, CloudVolume) or \
            isinstance(self.output_volume, OffsetArray)
        assert isinstance(self.exchange_storage, dict) or \
            isinstance(self.exchange_storage, Storage)
        assert isinstance(self.inference_engine, PatchInferenceEngine) or \
            isinstance(self.inference_engine, BlockInferenceEngine)

        # check the alignment of blocks
        if isinstance(self.output_volume, CloudVolume):
            for c, s, o in zip(
                    self.output_volume.info['scales'][0]['chunk_sizes'][0],
                    self.output_block_slices,
                    self.output_volume.voxel_offset):
                assert (s.start - o) % c == 0
                assert (s.stop - s.start) % c == 0

    def __call__(self, input_image):
        """
            run the inference
        args:
            input_image (OffsetArray): input image chunk with global offset
        """
        for i, o in zip(input_image.shape[-2:], self.output_buffer.shape[-2:]):
            assert i == o
        self.output_buffer = self.inference_engine(
            input_image, output_buffer=self.output_buffer)
        assert np.any(self.output_buffer > 0.0)
        self.chunk_manager.buffer_array = self.output_buffer
        self.chunk_manager.donate()
        self.chunk_manager.save_valid()


if __name__ == '__main__':
    params = InferenceAndDonateOptions().parse()

    # GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_ids)[1:-1]

    input_volume = CloudVolume(params.input_dir, fill_missing=True, cache=True)
    output_volume = CloudVolume(params.output_dir, non_aligned_writes=True)
    exchange_storage = Storage(params.exchange_dir)
    output_block_slices = params.output_block_slices

    if params.framework == 'pytorch':
        from frameworks.pytorch import PyTorchEngine
        patch_inference_engine = PyTorchEngine(params.model_path,
                                               params.net_path)
    elif params.framework == 'pznet':
        from frameworks.pznet import PZNetEngine
        patch_inference_engine = PZNetEngine(params.model_file_name,
                                             params.net_file_name)
    else:
        raise NotImplementedError('unknow framework backend of {}' %
                                  params.framework)

    block_inference_engine = BlockInferenceEngine(
        patch_inference_engine, params.patch_size, params.overlap,
        output_key=params.output_key, output_channels=params.output_channels)

    executor = InferenceAndDonate(input_volume, output_volume,
                                  exchange_storage,
                                  params.output_block_slices,
                                  params.overlap,
                                  block_inference_engine,
                                  output_channels=params.output_channels)

    # read input image
    input_slices = (slice(o.start, o.stop+v) for o, v in
                    zip(params.output_block_slices, params.overlap))
    input_offset = tuple(o.start for o in params.output_block_slices)
    input_image = input_volume[list(input_slices)[::-1]]
    input_image = np.transpose(input_image)
    input_image = np.squeeze(input_image, axis=0)
    input_image = np.ascontiguousarray(input_image, dtype='float32') / 255.0
    input_image = OffsetArray(input_image, global_offset=input_offset)

    # run inference
    print('running inference ...')
    executor(input_image)
