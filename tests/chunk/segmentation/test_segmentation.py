import numpy as np
from chunkflow.chunk.segmentation import Segmentation


def test_segmentation():
    arr = np.random.randint(256, size=(30,40,50), dtype=np.uint32)
    seg = Segmentation(arr, global_offset=(-1,-1,-1))

    arr2 = arr.astype(np.uint64)
    seg2 = Segmentation(arr2, global_offset=(-1,-1,-1))


    print('compare two segments.')
    score = seg.evaluate(seg2)
    print('evaluate score: ', score)
    assert score['voi_split'] == 0
    assert score['voi_merge'] == 0
    assert score['rand_split'] == 1 
    assert score['rand_merge'] == 1
