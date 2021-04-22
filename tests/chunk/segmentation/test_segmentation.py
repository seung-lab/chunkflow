import numpy as np
from chunkflow.chunk.segmentation import Segmentation


def test_segmentation():
    arr = np.random.randint(256, size=(30,40,50), dtype=np.uint32)
    seg = Segmentation(arr, voxel_offset=(-1,-1,-1))

    arr2 = arr.astype(np.uint64)
    seg2 = Segmentation(arr2, voxel_offset=(-1,-1,-1))


    print('compare two segments.')
    scores = seg.evaluate(seg2)
    print('evaluate scores: \n', scores)
    assert scores['variation_of_information'] == 0
    assert scores['rand_index'] == 1
    assert scores['adjusted_rand_index'] == 1 
    assert scores['fowlkes_mallows_index'] == 1
