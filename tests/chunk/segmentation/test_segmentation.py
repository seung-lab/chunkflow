import numpy as np
from chunkflow.chunk.segmentation import Segmentation


def test_segmentation():
    arr = np.random.randint(256, size=(3,4,5), dtype=np.uint32)
    seg = Segmentation(arr, global_offset=(-1,-1,-1))

    arr = arr.astype(np.uint64)
    seg = Segmentation(arr, global_offset=(-1,-1,-1))
