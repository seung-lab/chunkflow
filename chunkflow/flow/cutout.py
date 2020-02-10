import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
from cloudvolume.storage import Storage

from chunkflow.chunk.validate import validate_by_template_matching
from tinybrain import downsample_with_averaging
from chunkflow.chunk import Chunk
from .base import OperatorBase


class CutoutOperator(OperatorBase):
    def __init__(self,
                 volume_path: str,
                 mip: int = 0,
                 expand_margin_size=(0, 0, 0),
                 fill_missing: bool = False,
                 validate_mip: int = None,
                 blackout_sections: bool = None,
                 dry_run: bool = False,
                 name: str = 'cutout',
                 verbose: bool = True):
        super().__init__(name=name, verbose=verbose)
        self.volume_path = volume_path
        self.mip = mip
        self.expand_margin_size = expand_margin_size
        self.fill_missing = fill_missing
        self.validate_mip = validate_mip
        self.blackout_sections = blackout_sections
        self.dry_run = dry_run

        self.vol = CloudVolume(self.volume_path,
                               bounded=False,
                               fill_missing=self.fill_missing,
                               progress=self.verbose,
                               mip=self.mip,
                               parallel=False)

        if blackout_sections:
            with Storage(volume_path) as stor:
                self.blackout_section_ids = stor.get_json(
                    'blackout_section_ids.json')['section_ids']

        if self.validate_mip:
            self.validate_vol = CloudVolume(self.volume_path,
                                            bounded=False,
                                            fill_missing=self.fill_missing,
                                            progress=self.verbose,
                                            mip=self.validate_mip,
                                            parallel=False)

    def __call__(self, output_bbox):
        
        chunk_slices = tuple(
            slice(s.start - m, s.stop + m)
            for s, m in zip(output_bbox.to_slices(), self.expand_margin_size))
        
        if self.dry_run:
            input_bbox = Bbox.from_slices(chunk_slices)
            return Chunk.from_bbox(input_bbox)

        if self.verbose:
            print('cutout {} from {}'.format(chunk_slices[::-1],
                                             self.volume_path))

        # always reverse the indexes since cloudvolume use x,y,z indexing
        chunk = self.vol[chunk_slices[::-1]]
        # the cutout is fortran ordered, so need to transpose and make it C order
        chunk = chunk.transpose()
        # we can delay this transpose later
        # actually we do not need to make it contiguous
        # chunk = np.ascontiguousarray(chunk)

        # if the channel number is 1, squeeze it as 3d array
        # this should not be neccessary
        # TODO: remove this step and use 4D array all over this package.
        # always use 4D array will simplify some operations
        global_offset = tuple(s.start for s in chunk_slices)
        if chunk.shape[0] == 1:
            chunk = np.squeeze(chunk, axis=0)
        else:
            global_offset = (chunk.shape[0], ) + global_offset
        
        chunk = Chunk(chunk, global_offset=global_offset)

        if self.blackout_sections:
            chunk = self._blackout_sections(chunk)

        if self.validate_mip:
            self._validate_chunk(chunk)
        return chunk

    def _blackout_sections(self, chunk):
        """
        make some sections black.
        this was normally used for the section with bad alignment.
        The ConvNet was supposed to handle them better with black image.

        TODO: make this function as a separate operator
        """
        # current code only works with 3d image
        assert chunk.ndim == 3, "current code assumes that the chunk is 3D image."
        for z in self.blackout_section_ids:
            z0 = z - chunk.global_offset[0]
            if z0 >= 0 and z0 < chunk.shape[0]:
                chunk[z0, :, :] = 0
        return chunk

    def _validate_chunk(self, chunk):
        """
        check that all the input voxels was downloaded without black region  
        We have found some black regions in previous inference run, 
        so hopefully this will solve the problem.
        """
        if chunk.ndim == 4 and chunk.shape[0] > 1:
            chunk = chunk[0, :, :, :]

        chunk_mip = self.mip
        if self.verbose:
            print('validate chunk in mip {}'.format(self.validate_mip))
        assert self.validate_mip >= chunk_mip
        # only use the region corresponds to higher mip level
        # clamp the surrounding regions in XY plane
        # this assumes that the input dataset was downsampled starting from the
        # beginning offset in the info file
        global_offset = chunk.global_offset

        # factor3 follows xyz order in CloudVolume
        factor3 = np.array([
            2**(self.validate_mip - chunk_mip), 2
            **(self.validate_mip - chunk_mip), 1
        ],
                           dtype=np.int32)
        clamped_offset = tuple(go + f - (go - vo) % f for go, vo, f in zip(
            global_offset[::-1], self.vol.voxel_offset, factor3))
        clamped_stop = tuple(
            go + s - (go + s - vo) % f
            for go, s, vo, f in zip(global_offset[::-1], chunk.shape[::-1],
                                    self.vol.voxel_offset, factor3))
        clamped_slices = tuple(
            slice(o, s) for o, s in zip(clamped_offset, clamped_stop))
        clamped_bbox = Bbox.from_slices(clamped_slices)
        clamped_input = chunk.cutout(clamped_slices[::-1])
        # transform to xyz order
        clamped_input = np.transpose(clamped_input)
        # get the corresponding bounding box for validation
        validate_bbox = self.vol.bbox_to_mip(clamped_bbox,
                                             mip=chunk_mip,
                                             to_mip=self.validate_mip)
        #validate_bbox = clamped_bbox // factor3

        # downsample the input using avaraging
        # keep the z as it is since the mip only applies to xy plane
        # recursivly downsample the input
        # if we do it directly, the downsampled input will not be the same with the recursive one
        # because of the rounding error of integer division
        for _ in range(self.validate_mip - chunk_mip):
            clamped_input = downsample_with_averaging(clamped_input, (2, 2, 1))

        # validation by template matching
        assert validate_by_template_matching(clamped_input)

        validate_input = self.validate_vol[validate_bbox.to_slices()]
        if validate_input.shape[3] == 1:
            validate_input = np.squeeze(validate_input, axis=3)

        # use the validate input to check the downloaded input
        assert np.alltrue(validate_input == clamped_input)
