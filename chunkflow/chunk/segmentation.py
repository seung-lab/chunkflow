__doc__ = """Image chunk class"""
import logging
import multiprocessing
from typing import Union

import numpy as np
from .base import Chunk

# from ...lib.gala import evaluate
from chunkflow.lib.gala import evaluate

import kimimaro
import fastremap

from cloudvolume.storage import Storage


class Segmentation(Chunk):
    """
    a chunk of segmentation volume.
    """
    def __init__(self, array, voxel_offset=None):
        super().__init__(array, voxel_offset=voxel_offset)
        assert array.ndim == 3
        assert np.issubdtype(array.dtype, np.integer)

    @classmethod
    def from_chunk(cls, chunk):
        assert isinstance(chunk, Chunk)
        return cls(chunk.array, voxel_offset=chunk.voxel_offset)

    def evaluate(self, groundtruth, size_threshold: int=1000):
        """
        Parameters:
            size_threshold [int]: size threshold for Edit Distance. 
                Ignore splits or merges smaller than this number of voxels.
        """
        if not np.issubdtype(self.dtype, np.uint64):
            this = self.astype(np.uint64)
        else:
            this = self

        if not np.issubdtype(groundtruth.dtype, np.uint64):
            groundtruth = groundtruth.astype(np.uint64)
        
        if isinstance(groundtruth, Chunk):
            groundtruth = groundtruth.array

        rand_index = evaluate.rand_index(this.array, groundtruth)
        adjusted_rand_index = evaluate.adj_rand_index(this.array, groundtruth)
        variation_of_information = evaluate.vi(this.array, groundtruth)
        fowlkes_mallows_index = evaluate.fm_index(this.array, groundtruth)
        edit_distance = evaluate.edit_distance(this.array, groundtruth, size_threshold=size_threshold)
        print('rand index: ', rand_index)
        print('adjusted rand index: ', adjusted_rand_index)
        print('variation of information: ', variation_of_information)
        print('edit distance: ', edit_distance)
        print('Fowlkes Mallows Index: ', fowlkes_mallows_index)

        ret = {}
        ret['rand_index'] = rand_index
        ret['adjusted_rand_index'] = adjusted_rand_index
        ret['variation_of_information'] = variation_of_information
        ret['fowlkes_mallows_index'] = fowlkes_mallows_index
        ret['edit_distance'] = edit_distance
        return ret

    def remap(self, start_id: int):
        fastremap.renumber(self.array, preserve_zero=True, in_place=True)
        seg = self.astype(np.uint64)
        seg.array[seg.array>0] += start_id
        start_id = seg.max()
        return seg, start_id

    def mask_fragments(self, voxel_num_threshold: int):
        uniq, counts = fastremap.unique(self.array, return_counts=True)
        fragment_ids = uniq[counts <= voxel_num_threshold]
        logging.info(f'masking out {len(fragment_ids)} fragments in {len(uniq)} with a percentage of {len(fragment_ids)/len(uniq)}')
        self.array = fastremap.mask(self.array, fragment_ids)

    def mask_except(self, selected_obj_ids: Union[str, list, set]):
        if selected_obj_ids is None:
            logging.warning('we have not selected any objects to mask out.')
            return

        if isinstance(selected_obj_ids, str) and selected_obj_ids.endswith('.json'):
            # assume that ids is a json file in the storage path
            json_storage = Storage(os.path.dirname(selected_obj_ids))
            ids_str = json_storage.get_file(os.path.basename(selected_obj_ids))
            selected_obj_ids = set(json.loads(ids_str))
            assert len(selected_obj_ids) > 0
            logging.info(f'number of selected objects: {len(selected_obj_ids)}')
        elif isinstance(selected_obj_ids, str):
            # a simple string, like "34,45,56,23"
            # this is used for small object numbers
            selected_obj_ids = set([int(id) for id in selected_obj_ids.split(',')])
        
        self.array = fastremap.mask_except(self.array, list(selected_obj_ids))


    def skeletonize(self, voxel_size):
        skels = kimimaro.skeletonize(
            self.array,
            anisotropy=voxel_size,
            parallel=multiprocessing.cpu_count() // 2
        )
        return skels

