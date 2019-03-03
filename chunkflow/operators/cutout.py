import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox

from chunkflow.lib.validate import validate_by_template_matching
from chunkflow.igneous.downsample import downsample_with_averaging
from chunkflow.chunk import Chunk
from .operator_base import OperatorBase


class CutoutOperator(OperatorBase):
    def __init__(self, volume_path, mip=0, 
                 expand_margin_size=(0,0,0),
                 verbose=True, fill_missing=False,
                 validate_mip=None, name='cutout'):
        super().__init__(name=name)
        self.volume_path = volume_path
        self.mip = mip
        self.expand_margin_size = expand_margin_size
        self.verbose = verbose
        self.fill_missing = fill_missing
        self.validate_mip = validate_mip
        
        self.vol = CloudVolume(
            self.volume_path,
            bounded=False,
            fill_missing=self.fill_missing,
            progress=self.verbose,
            mip=self.mip,
            parallel=False)

        if self.validate_mip:
            self.validate_vol = CloudVolume(
                self.volume_path,
                bounded=False,
                fill_missing=self.fill_missing,
                progress=self.verbose,
                mip=self.validate_mip,
                parallel=False)


    def __call__(self, output_bbox):
    
        chunk_slices = tuple(slice(s.start - m, s.stop + m) 
                             for s, m in zip(output_bbox.to_slices(), 
                                             self.expand_margin_size))
        if self.verbose:
            print('cutout {} from {}'.format(
                chunk_slices[::-1], self.volume_path))

        # always reverse the indexes since cloudvolume use x,y,z indexing
        chunk = self.vol[chunk_slices[::-1]]
        # the cutout is fortran ordered, so need to transpose and make it C order
        chunk = np.transpose(chunk)
        chunk = np.ascontiguousarray(chunk)
        # if the channel number is 1, squeeze it as 3d array
        # this should not be neccessary 
        # TODO: remove this step and use 4D array all over this package.
        # always use 4D array will simplify some operations
        chunk = np.squeeze(chunk, axis=0)
        global_offset = tuple(s.start for s in chunk_slices)

        chunk = Chunk(chunk, global_offset=global_offset)
       
        if self.validate_mip:
            self._validate_chunk(chunk)
        return chunk

    def _validate_chunk(self, chunk):
        """
        check that all the input voxels was downloaded without black region  
        We have found some black regions in previous inference run, 
        so hopefully this will solve the problem.
        """
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
            2**(self.validate_mip - chunk_mip), 
            2**(self.validate_mip - chunk_mip), 1], dtype=np.int32)
        clamped_offset = tuple(go + f - (go - vo) % f for go, vo, f in zip(
            global_offset[::-1], self.vol.voxel_offset, factor3))
        clamped_stop = tuple(go + s - (go + s - vo) % f
                             for go, s, vo, f in zip(
                                 global_offset[::-1], chunk.shape[::-1],
                                 self.vol.voxel_offset, factor3))
        clamped_slices = tuple(
            slice(o, s) for o, s in zip(clamped_offset, clamped_stop))
        clamped_bbox = Bbox.from_slices(clamped_slices)
        clamped_input = chunk.cutout(clamped_slices[::-1])
        # transform to xyz order
        clamped_input = np.transpose(clamped_input)
        # get the corresponding bounding box for validation
        validate_bbox = self.vol.bbox_to_mip(
            clamped_bbox, mip=chunk_mip, to_mip=self.validate_mip)
        #validate_bbox = clamped_bbox // factor3

        # downsample the input using avaraging
        # keep the z as it is since the mip only applies to xy plane
        # recursivly downsample the input
        # if we do it directly, the downsampled input will not be the same with the recursive one
        # because of the rounding error of integer division
        for _ in range(self.validate_mip - chunk_mip):
            clamped_input = downsample_with_averaging(
                clamped_input, np.array([2, 2, 1], dtype=np.int32))

        # validation by template matching
        assert validate_by_template_matching(clamped_input)

        validate_input = self.validate_vol[validate_bbox.to_slices()]
        assert validate_input.shape[3] == 1
        validate_input = np.squeeze(validate_input, axis=3)

        # use the validate input to check the downloaded input
        assert np.alltrue(validate_input == clamped_input)
