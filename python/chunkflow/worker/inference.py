import numpy as np
# from .offset_array import OffsetArray
from .aligned_patch_inference import AlignedPatchInference
from enum import IntEnum


class Role(IntEnum):
    Donor = 1
    Waster = 0
    Receiver = -1


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
        self.donation_map = params.get('donation_map',
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

    def _evaluate_boundary_chunks(self):
        # initialized as no dependent chunks
        self.receiving_chunk_ranges_list = []
        self.donating_chunk_ranges_list = []
        self.input_chunk_ranges = [range(o.start-m, o.stop+m)
                                   for (o, m) in zip(self.output_ranges,
                                                     self.shared_margin_size)]

        # iterate through chunk faces
        if self.donation_map['z'][0] == Role.Receiver:
            chunk_ranges = (
                range(self.output_ranges[0].start,
                      self.output_ranges[0].start+self.shared_margin_size[0]),
                self.inner_chunk_ranges[1], self.inner_chunk_ranges[2])
            self.receiving_chunk_ranges_list.append(chunk_ranges)
            self.input_chunk_ranges[0] = range(
                self.input_chunk_ranges[0].start +
                self.patch_stride_by_voxel[0],
                self.input_chunk_ranges[0].stop)
        elif self.donation_map['z'][0] == Role.Donor:
            chunk_ranges = (
                range(self.output_ranges[0].start-self.shared_margin_size[0],
                      self.output_ranges[0].start),
                self.inner_chunk_ranges[1], self.inner_chunk_ranges[2])
            self.donating_chunk_ranges_list.append(chunk_ranges)

        if self.donation_map['z'][1] == Role.Receiver:
            chunk_ranges = (
                range(self.output_ranges[0].stop-self.shared_margin_size[0],
                    self.output_ranges[0].stop),
                self.inner_chunk_ranges[1], self.inner_chunk_ranges[2])
            self.receiving_chunk_ranges_list.append(chunk_ranges)
            self.input_chunk_ranges[0] = range(
                self.input_chunk_ranges[0].start,
                self.input_chunk_ranges[0].stop-self.patch_stride_by_voxel[0])
        elif self.donation_map['z'][1] == Role.Donor:
            chunk_ranges = (
                range(self.output_ranges[0].stop,
                    self.output_ranges[0].stop+self.shared_margin_size[0]),
                self.inner_chunk_ranges[1], self.inner_chunk_ranges[2])
            self.donating_chunk_ranges_list.append(chunk_ranges)

        if self.donation_map['y'][0] == Role.Receiver:
            chunk_ranges = (
                self.inner_chunk_ranges[0],
                range(self.output_ranges[1].start,
                      self.output_ranges[1].start+self.shared_margin_size[1]),
                self.inner_chunk_ranges[2])

            self.receiving_chunk_ranges_list.append(chunk_ranges)
            self.input_chunk_ranges[1] = range(
                self.input_chunk_ranges[1].start +
                self.patch_stride_by_voxel[1],
                self.input_chunk_ranges[1].stop)
        elif self.donation_map['y'][0] == Role.Donor:
            chunk_ranges = (
                self.inner_chunk_ranges[0],
                range(self.output_ranges[1].start-self.shared_margin_size[1],
                      self.output_ranges[1].start),
                self.inner_chunk_ranges[2])

            self.donating_chunk_ranges_list.append(chunk_ranges)

        if self.donation_map['y'][1] == Role.Receiver:
            chunk_ranges = (
                self.inner_chunk_ranges[0],
                range(self.output_ranges[1].stop-self.shared_margin_size[1],
                      self.output_ranges[1].stop),
                self.inner_chunk_ranges[2])
            self.receiving_chunk_ranges_list.append(chunk_ranges)
            self.input_chunk_ranges[1] = range(
                self.input_chunk_ranges[1].start,
                self.input_chunk_ranges[1].stop -
                self.patch_stride_by_voxel[1])
        elif self.donation_map['y'][1] == Role.Donor:
            chunk_ranges = (
                self.inner_chunk_ranges[0],
                range(self.output_ranges[1].stop,
                      self.output_ranges[1].stop+self.shared_margin_size[1]),
                self.inner_chunk_ranges[2])
            self.donating_chunk_ranges_list.append(chunk_ranges)

        if self.donation_map['x'][0] == Role.Receiver:
            chunk_ranges = (
                self.inner_chunk_ranges[0], self.inner_chunk_ranges[1],
                range(self.output_ranges[2].start,
                      self.output_ranges[2].start+self.shared_margin_size[2]))
            self.receiving_chunk_ranges_list.append(chunk_ranges)
            self.input_chunk_ranges[2] = range(
                self.input_chunk_ranges[2].start +
                self.patch_stride_by_voxel[2],
                self.input_chunk_ranges[2].stop)
        elif self.donation_map['x'][0] == Role.Donor:
            chunk_ranges = (
                self.inner_chunk_ranges[0], self.inner_chunk_ranges[1],
                range(self.output_ranges[2].start-self.shared_margin_size[2],
                      self.output_ranges[2].start))
            self.donating_chunk_ranges_list.append(chunk_ranges)

        if self.donation_map['x'][1] == Role.Receiver:
            chunk_ranges = (
                self.inner_chunk_ranges[0], self.inner_chunk_ranges[1],
                range(self.output_ranges[2].stop-self.shared_margin_size[2],
                      self.output_ranges[2].stop))
            self.receiving_chunk_ranges_list.append(chunk_ranges)
            self.input_chunk_ranges[2] = range(
                self.input_chunk_ranges[2].start,
                self.input_chunk_ranges[2].stop -
                self.patch_stride_by_voxel[2])
        elif self.donation_map['x'][1] == Role.Donor:
            chunk_ranges = (
                self.inner_chunk_ranges[0], self.inner_chunk_ranges[1],
                range(self.output_ranges[2].stop,
                      self.output_ranges[2].stop+self.shared_margin_size[2]))
            self.donating_chunk_ranges_list.append(chunk_ranges)

        # iterate through edges
        # note that the input chunk range was already adjusted,
        # no need to change anymore
        if self.donation_map['z'][0] == Role.Receiver or \
                self.donation_map['y'][0] == Role.Receiver:
            chunk_ranges = (
                range(self.output_ranges[0].start,
                      self.output_ranges[0].start+self.shared_margin_size[0]),
                range(self.output_ranges[1].start,
                      self.output_ranges[1].start+self.shared_margin_size[1]),
                self.inner_chunk_ranges[2])
            self.receiving_chunk_ranges_list.append(chunk_ranges)
        elif self.donation_map['z'][0] == Role.Donor and \
                self.donation_map['y'][0] == Role.Donor:
            chunk_ranges = (
                range(self.output_ranges[0].start-self.shared_margin_size[0],
                      self.output_ranges[0].start),
                range(self.output_ranges[1].start-self.shared_margin_size[1],
                      self.output_ranges[1].start),
                self.inner_chunk_ranges[2])
            self.donating_chunk_ranges_list.append(chunk_ranges)

        if self.donation_map['z'][1] == Role.Receiver or \
                self.donation_map['y'][0] == Role.Receiver:
            chunk_ranges = (
                range(self.output_ranges[0].stop-self.shared_margin_size[0],
                      self.output_ranges[0].stop),
                range(self.output_ranges[1].start,
                      self.output_ranges[1].start+self.shared_margin_size[1]),
                self.inner_chunk_ranges[2])
            self.receiving_chunk_ranges_list.append(chunk_ranges)
        elif self.donation_map['z'][1] == Role.Donor and \
                self.donation_map['y'][0] == Role.Donor:
            chunk_ranges = (
                range(self.output_ranges[0].stop,
                      self.output_ranges[0].stop+self.shared_margin_size[0]),
                range(self.output_ranges[1].start-self.shared_margin_size[1],
                      self.output_ranges[1].start),
                self.inner_chunk_ranges[2])
            self.donating_chunk_ranges_list.append(chunk_ranges)

        if self.donation_map['z'][0] == Role.Receiver or \
                self.donation_map['y'][1] == Role.Receiver:
            chunk_ranges = (
                range(self.output_ranges[0].start,
                      self.output_ranges[0].start+self.shared_margin_size[0]),
                range(self.output_ranges[1].stop-self.shared_margin_size[1],
                      self.output_ranges[1].stop),
                self.inner_chunk_ranges[2])
            self.receiving_chunk_ranges_list.append(chunk_ranges)
        elif self.donation_map['z'][0] == Role.Donor and \
                self.donation_map['y'][1] == Role.Donor:
            chunk_ranges = (
                range(self.output_ranges[0].start-self.shared_margin_size[0],
                      self.output_ranges[0].start),
                range(self.output_ranges[1].stop,
                      self.output_ranges[1].stop+self.shared_margin_size[1]),
                self.inner_chunk_ranges[2])
            self.donating_chunk_ranges_list.append(chunk_ranges)

        if self.donation_map['z'][1] == Role.Receiver or \
                self.donation_map['y'][1] == Role.Receiver:
            chunk_ranges = (
                range(self.output_ranges[0].stop-self.shared_margin_size[0],
                      self.output_ranges[0].stop),
                range(self.output_ranges[1].stop-self.shared_margin_size[1],
                      self.output_ranges[1].stop),
                self.inner_chunk_ranges[2])
            self.receiving_chunk_ranges_list.append(chunk_ranges)
        elif self.donation_map['z'][1] == Role.Donor and \
                self.donation_map['y'][1] == Role.Donor:
            chunk_ranges = (
                range(self.output_ranges[0].stop,
                      self.output_ranges[0].stop+self.shared_margin_size[0]),
                range(self.output_ranges[1].stop,
                      self.output_ranges[1].stop+self.shared_margin_size[1]),
                self.inner_chunk_ranges[2])
            self.donating_chunk_ranges_list.append(chunk_ranges)

        # iterate through corners



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
