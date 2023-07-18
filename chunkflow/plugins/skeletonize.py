#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing

import kimimaro
from chunkflow.chunk import Chunk

from cloudfiles import CloudFiles

def skeletonize(seg: Chunk, voxel_size):
    skels = kimimaro.skeletonize(
        seg.array,
        anisotropy=voxel_size,
        parallel=multiprocessing.cpu_count() // 2
    )
    return skels

def execute(seg: Chunk, voxel_size: tuple, output_path: str=None):
    """Create mesh files from segmentation."""
    storage = CloudFiles(output_path)

    if seg is None:
        print('no segmentation, skip computation.')
        return None

    print('skeletonize segmentation...')
    
    skels = skeletonize(seg, voxel_size)
    bbox_str = seg.bbox.string
    for neuron_id, skel in skels.items():
        file_name = f'{neuron_id}:{bbox_str}'
        storage.put(file_name, skel.to_precomputed())
    return skels 
