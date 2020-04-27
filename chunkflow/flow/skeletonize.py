#!/usr/bin/env python
# -*- coding: utf-8 -*-


from .base import OperatorBase
from chunkflow.chunk.segmentation import Segmentation

from cloudvolume.storage import Storage


class SkeletonizeOperator(OperatorBase):
    """Create mesh files from segmentation."""
    def __init__(self, output_path,
                 name: str='skeletonize',
                 verbose: int=1):
        """
        Parameters
        ------------
        output_path:
            where to put the skeleton files
        name: 
            operator name.
        verbose:
            print out informations or not.
        """
        super().__init__(name=name, verbose=verbose)
        self.storage = Storage(output_path)

    def __call__(self, seg, voxel_size):
        if seg is None:
            print('no segmentation, skip computation.')
            return None
 
        if self.verbose:
            print('skeletonize segmentation...')
      
        seg = Segmentation.from_chunk(seg)
        skels = seg.skeletonize(voxel_size)
        bbox_str = seg.bbox.to_filename()
        for neuron_id, skel in skels.items():
            file_name = f'{neuron_id}:{bbox_str}'
            self.storage.put_file(file_name, skel.to_precomputed())
        return skels 
