import logging

import numpy as np
from cloudvolume import CloudVolume
from cloudfiles import CloudFiles

from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.chunk.validate import validate_by_template_matching
from tinybrain import downsample_with_averaging
from chunkflow.chunk import Chunk
from .base import OperatorBase


class LoadPrecomputedOperator(OperatorBase):
    def __init__(self,
                 volume_path: str,
                 mip: int = 0,
                 expand_margin_size: Cartesian=Cartesian(0, 0, 0),
                 expand_direction: int = None,
                 fill_missing: bool = False,
                 validate_mip: int = None,
                 blackout_sections: bool = None,
                 use_https: bool = False,
                 dry_run: bool = False,
                 name: str = 'cutout'):
        super().__init__(name=name)
        self.volume_path = volume_path
        self.mip = mip
        self.fill_missing = fill_missing
        self.validate_mip = validate_mip
        self.blackout_sections = blackout_sections
        self.dry_run = dry_run

        if isinstance(expand_margin_size, tuple):
            expand_margin_size = Cartesian.from_collection(expand_margin_size)

        if expand_direction == 1:
            expand_margin_size = (0, 0, 0, *expand_margin_size)
        elif expand_direction == -1:
            expand_margin_size = (*expand_margin_size, 0, 0, 0)
        else: 
            assert expand_direction is None
        self.expand_margin_size = expand_margin_size
        
        if blackout_sections:
            stor = CloudFiles(volume_path)
            self.blackout_section_ids = stor.get_json(
                'blackout_section_ids.json')['section_ids']

        verbose = (logging.getLogger().getEffectiveLevel() <= 30)
        self.vol = CloudVolume(
            self.volume_path,
            bounded=False,
            fill_missing=self.fill_missing,
            progress=verbose,
            mip=self.mip,
            cache=False,
            use_https=use_https,
            green_threads=True)
            #parallel=True,
        
    def __call__(self, output_bbox: BoundingBox):
        # if we do not clone this bounding box, 
        # the bounding box in task will be modified!
        assert isinstance(output_bbox, BoundingBox)
        output_bbox = output_bbox.clone()
        output_bbox = output_bbox.adjust(self.expand_margin_size)
        chunk_slices = output_bbox.slices
        
        if self.dry_run:
            # input_bbox = BoundingBox.from_slices(chunk_slices)
            # we can not use pattern=zero since it might got skipped by 
            # the operator of skip-all-zero
            return Chunk.from_bbox(
                output_bbox,
                pattern='random',
                dtype=self.vol.dtype,
                voxel_size=Cartesian.from_collection(self.vol.resolution[::-1]),
            )

        logging.info('cutout {} from {}'.format(chunk_slices[::-1],
                                             self.volume_path))

        # always reverse the indexes since cloudvolume use x,y,z indexing
        chunk = self.vol[chunk_slices[::-1]]
        chunk = np.asarray(chunk)
        # the cutout is fortran ordered, so need to transpose and make it C order
        chunk = chunk.transpose()

        # we can delay this transpose later
        # actually we do not need to make it contiguous
        # chunk = np.ascontiguousarray(chunk)

        # if the channel number is 1, squeeze it as 3d array
        # this should not be neccessary
        # TODO: remove this step and use 4D array all over this package.
        # always use 4D array will simplify some operations
        # voxel_offset = Cartesian(s.start for s in chunk_slices)
        if chunk.shape[0] == 1:
            chunk = np.squeeze(chunk, axis=0)
        
        chunk = Chunk(
            chunk, 
            voxel_offset=output_bbox.start,
            voxel_size=Cartesian.from_collection(self.vol.resolution[::-1]),
            layer_type=self.vol.layer_type)

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
            z0 = z - chunk.voxel_offset[0]
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
        
        validate_vol = CloudVolume(self.volume_path,
                                   bounded=False,
                                   fill_missing=self.fill_missing,
                                   progress=False,
                                   mip=self.validate_mip,
                                   cache=False,
                                   green_threads=True)


        chunk_mip = self.mip
        logging.info('validate chunk in mip {}'.format(self.validate_mip))
        assert self.validate_mip >= chunk_mip
        # only use the region corresponds to higher mip level
        # clamp the surrounding regions in XY plane
        # this assumes that the input dataset was downsampled starting from the
        # beginning offset in the info file
        voxel_offset = chunk.voxel_offset

        # factor3 follows xyz order in CloudVolume
        factor3 = np.array([
            2**(self.validate_mip - chunk_mip), 2
            **(self.validate_mip - chunk_mip), 1
        ],
                           dtype=np.int32)
        clamped_offset = tuple(go + f - (go - vo) % f for go, vo, f in zip(
            voxel_offset[::-1], self.vol.voxel_offset, factor3))
        clamped_stop = tuple(
            go + s - (go + s - vo) % f
            for go, s, vo, f in zip(voxel_offset[::-1], chunk.shape[::-1],
                                    self.vol.voxel_offset, factor3))
        clamped_slices = tuple(
            slice(o, s) for o, s in zip(clamped_offset, clamped_stop))
        clamped_bbox = BoundingBox.from_slices(clamped_slices)
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

        validate_input = validate_vol[validate_bbox.slices]
        if validate_input.shape[3] == 1:
            validate_input = np.squeeze(validate_input, axis=3)

        # use the validate input to check the downloaded input
        assert np.alltrue(validate_input == clamped_input)
