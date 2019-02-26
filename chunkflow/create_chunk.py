import numpy as np


def create_chunk(size=(64, 64, 64), dtype='uint8'):
    def _make_black_box():
        slices = tuple(slice(s//4, -s//4) for s in size)
        chunk[slices] = 0

    if np.issubdtype(dtype, np.floating):
        chunk = np.random.rand(*size).astype(dtype)
        _make_black_box()
        return chunk
    elif dtype=='uint8':
        chunk = np.random.randint(0, 256, size=size, dtype=dtype)
        _make_black_box()
        return chunk
    elif np.issubdtype(dtype, np.integer):
        raise NotImplementedError()
    else:
        raise NotImplementedError()
