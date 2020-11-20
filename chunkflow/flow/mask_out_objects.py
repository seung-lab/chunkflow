import logging
import os
import json
import numpy as np

import fastremap

from cloudvolume.storage import Storage

from chunkflow.chunk import Chunk
from .base import OperatorBase


class MaskOutObjectsOperator(OperatorBase):
    """Create mesh files from segmentation."""
    def __init__(self, dust_size_threshold: int, selected_obj_ids: str,
                 name: str='mask-out-objects'):
        """
        Parameters
        ------------
        dust_threshold:
            do not mesh tiny objects with voxel number less than threshold
        ids:
            only mesh the selected segmentation ids, other segments will not be meshed.
        manifest:
            create manifest files or not. This should 
            not be True if you are only doing meshing for a segmentation chunk.
        name: 
            operator name.

        Note that some functions are adopted from igneous.
        """
        super().__init__(name=name)
        self.dust_size_threshold = dust_size_threshold
        
        if selected_obj_ids:
            if selected_obj_ids.endswith('.json'):
                # assume that ids is a json file in the storage path
                json_storage = Storage(os.path.dirname(selected_obj_ids))
                ids_str = json_storage.get_file(os.path.basename(selected_obj_ids))
                self.ids = set(json.loads(ids_str))
                assert len(self.ids) > 0
                logging.info(f'number of selected objects: {len(self.ids)}')
            else:
                # a simple string, like "34,45,56,23"
                # this is used for small object numbers
                self.ids = set([int(id) for id in selected_obj_ids.split(',')])

    def _remove_dust(self, seg: np.ndarray):
        """
        this function is adopted from igneous.
        """
        logging.info('remove dust segments')

        if self.dust_size_threshold or self.ids:
            segids, voxel_nums = np.unique(seg, return_counts=True)
            dust_segids = [sid for sid, ct in
                           zip(segids, voxel_nums) 
                           if ct < self.dust_size_threshold]

            seg = fastremap.mask(seg, dust_segids, in_place=True)
        return seg

    def _only_keep_selected(self, seg: np.ndarray):
        logging.info('only keep selected segment ids, and remove others.')

        # precompute the remap function
        if self.ids:
            #segids = set(np.unique(seg))
            #ids = self.ids.intersection( segids )
            seg = fastremap.mask_except(seg, list(self.ids), in_place=True)
        return seg
    
    def __call__(self, seg: Chunk):
        """Mask out selected objects in the segmentation chunk.
        
        Parameters
        ------------
        seg:
            3D segmentation chunk.
        """
        assert isinstance(seg, Chunk)
        assert seg.ndim == 3
        assert np.issubdtype(seg.dtype, np.integer)
        
        voxel_offset = seg.voxel_offset
        # use ndarray after getting the bounding box
        seg = seg.array

        seg = self._only_keep_selected(seg)
        if np.alltrue(seg == 0):
            logging.info('no segmentation id is selected!')
            return
        seg = self._remove_dust(seg)
        
        return Chunk(seg, voxel_offset=voxel_offset)
