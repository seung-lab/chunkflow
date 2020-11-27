
from chunkflow.chunk import Chunk
from .base import OperatorBase

import numpy as np
import fastremap


def remap_segmentation(seg: Chunk, start_id: int):
    fastremap.renumber(seg.array, preserve_zero=True, in_place=True)
    seg = seg.astype(np.uint64)
    seg.array[seg.array>0] += start_id
    start_id = seg.max()
    return seg, start_id