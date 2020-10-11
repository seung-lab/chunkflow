#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from chunkflow.chunk.affinity_map import AffinityMap

def test_affinity_map_construction():
    arr = np.random.rand(3,3,4,5).astype(np.float32)
    aff = AffinityMap(arr, voxel_offset=(0, -1,-1,-1))
 
