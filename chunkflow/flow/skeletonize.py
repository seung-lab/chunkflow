#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

from .base import OperatorBase
from chunkflow.chunk.segmentation import Segmentation

from cloudfiles import CloudFiles

class SkeletonizeOperator(OperatorBase):
    """Create mesh files from segmentation."""
    def __init__(self, output_path,
                 name: str='skeletonize'):
        """
        Parameters
        ------------
        output_path:
            where to put the skeleton files
        name: 
            operator name.
        """
        super().__init__(name=name)
        self.storage = CloudFiles(output_path)

    def __call__(self, seg, voxel_size):
        if seg is None:
            print('no segmentation, skip computation.')
            return None
 
        logging.info('skeletonize segmentation...')
      
        seg = Segmentation.from_chunk(seg)
        skels = seg.skeletonize(voxel_size)
        bbox_str = seg.bbox.string
        for neuron_id, skel in skels.items():
            file_name = f'{neuron_id}:{bbox_str}'
            self.storage.put(file_name, skel.to_precomputed())
        return skels 
