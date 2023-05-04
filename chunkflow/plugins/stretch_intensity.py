import numpy as np


from chunkflow.chunk import Chunk


def execute(chk: Chunk, min_threshold: int = 0, max_threshold: int = 256):
    assert np.issubdtype(chk.dtype, np.uint8)
    assert min_threshold>=0
    assert max_threshold <= 256
    assert min_threshold < max_threshold

    lookup_table = np.arange(256, dtype=np.float32)
    lookup_table -= min_threshold
    lookup_table /= (max_threshold - min_threshold)
    lookup_table[lookup_table<0.] = 0.
    lookup_table[lookup_table>1.] = 1.
    lookup_table *= 255
    lookup_table = lookup_table.astype(np.uint8)

    chk.array = lookup_table[chk.array]
    return chk

