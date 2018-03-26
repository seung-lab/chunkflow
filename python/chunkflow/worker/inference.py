#!/usr/bin/env python

import os
import numpy as np
# from .offset_array import OffsetArray
from .aligned_patch_inference import AlignedPatchInference
from .chunk_manager import ChunkManager, Role
from .options import InferenceOptions


class InferenceTask(object):
    """
        InferenceTask
    """
    def __init__(self):
        pass

    def __call__(self, params):
        # setup parameters
        self._setup(params)

        # initialize the output block as buffer, all other patches will be
        # blended into this buffer
        self.output_block = np.zeros(self.output_aligned_block_size,
                                     dtype=np.float32)

        self._blend_dependent_chunks()
        self._blend_convnet_output_chunk()

        return self.output_block

    def _setup(self, params):
        """
        params:
            params: dict, contains the setup parameters. the keys should be:
                grid_index, output_chunk_size,
                output_global_offset, patch_size, patch_stride,
                input_data_layer_path, output_data_layer_path,
                inference_engine, patch_stride, output_key
        note that the inference_engine should be setup
        """
        # self.convnet_input_key = params['convnet_input_key']
        self.grid_index = params['grid_index']
        self.output_ranges = params['output_ranges']
        self.patch_size = params['patch_size']
        self.input_data_layer_path = params['input_data_layer_path']
        self.output_data_layer_path = params['output_data_layer_path']
        self.donated_chunk_path = params['donated_chunk_path']

        self.patch_stride_percentile = params['patch_stride_percentile']

        # 1:donate, -1:receive, 0:do not donate
        # in the boundary, there is no need to donate since there is no task
        # need them.
        self.role_mask = params.get('role_mask',
                                       dict(x=(Role.Waster, Role.Waster),
                                            y=(Role.Waster, Role.Waster),
                                            z=(Role.Waster, Role.Waster)))

        assert np.less_equal(self.patch_stride_percentile, 1.0)
        # this was formated as percentile, transform to voxel number
        self.patch_stride_by_voxel = tuple(np.round(np.multiply(
            self.patch_size, self.patch_stride_percentile)))
        # output block size should be dividable by stride
        assert not np.any(np.mod(self.output_block_size,
                                 self.patch_stride_by_voxel))

        self.patch_overlap = np.subtract(self.patch_size,
                                         self.patch_stride_by_voxel)
        self.shared_margin_size = np.divide(self.patch_size, 2)

        self.aligned_patch_inference = AlignedPatchInference(params)

    @property
    def output_global_offset(self):
        return (x.start for x in self.output_ranges)

    @property
    def output_block_size(self):
        return (x.stop-x.start for x in self.output_ranges)

    @property
    def inner_chunk_ranges(self):
        """the inner chunk range in the case of getting all the donation"""
        return (range(o.start+m, o.stop-m) for (o, m) in
                zip(self.output_ranges, self.shared_margin_size))


    def _blend_dependent_chunks(self):
        dependent_chunk_range_list = self.dependent_chunk_range_list()



    def _blend_convnet_output_chunk(self):
        NotImplementedError()

    def _get_input_chunk(self):
        NotImplementedError()

    def _is_donor(self, grid_index ):
        return grid_index % 2 == 0

    def _is_receiver(self, grid_index):
        return grid_index % 2 == 1


if __name__ == '__main__':
    params = InferenceOptions().parse()

    # GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(params.gpu_ids)

    # run inference
    print('running inference ...')
    inference_task = InferenceTask()
    inference_task(params)
