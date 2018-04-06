#!/usr/bin/env python
import numpy as np
from offset_array import OffsetArray
from cloudvolume.cloudvolume import CloudVolume
from cloudvolume.storage import Storage
from chunk_manager import ChunkManager
from options import ReceiveAndBlendOptions


class ReceiveAndBlend(object):
    """
        ReceiveAndBlend
    receive the sharing chunks, blend them and then save to output volume.
    args:
        output_volume: similar with input_volume
        exchange_storage (key-value store or dict): exchange the shared chunks.
        output_block_slices (tuple of slices): have to be aligned with output \
            CloudVolume block size
        overlap (tuple of int): the margin size blending with neighboring \
            patches/blocks.
    """
    def __init__(self, output_volume, exchange_storage, output_block_slices,
                 overlap):
        self.output_volume = output_volume
        self.exchange_storage = exchange_storage
        self.output_block_slices = output_block_slices
        self.overlap = overlap

        output_buffer_size = (s.stop-s.start+o for s, o in
                              zip(output_block_slices, overlap))
        output_buffer_offset = (s.start for s in output_block_slices)
        self.output_buffer = OffsetArray(np.zeros(output_buffer_size),
                                         global_offset=output_buffer_offset)

        self.chunk_manager = ChunkManager(self.output_buffer, exchange_storage,
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
            for c, s, o in zip(self.output_volume.chunk_size,
                               self.output_block_slices,
                               self.output_volume.voxel_offset):
                assert (s.start - o) % c == 0
                assert (s.stop - s.start) % c == 0


    def __call__(self):
        self.chunk_manager.receive_and_blend()


if __name__ == '__main__':
    params = ReceiveAndBlendOptions().parse()

    input_volume = CloudVolume(params.input_dir, fill_missing=True, cache=True)
    output_volume = CloudVolume(params.output_dir)
    exchange_storage = Storage(params.exchange_dir)
    output_block_slices = params.output_block_slices

    output_buffer_size = (s.stop-s.start+o for s, o in
                            zip(output_block_slices, overlap))
    output_buffer_offset = (s.start for s in output_block_slices)
    output_buffer = OffsetArray(np.zeros(output_buffer_size),
                                global_offset=output_buffer_offset)

    chunk_manager = ChunkManager(output_buffer, exchange_storage, output_block_slices,
                                 overlap)

    chunk_manager.receive_and_blend()
