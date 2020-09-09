import numpy as np

def exec(chunk: np.ndarray):
    assert np.issubdtype(chunk.dtype, np.float32)
    assert chunk.min() >= -1
    assert chunk.max() <= 1
    chunk = (chunk + 1.0) / 2.0
    return chunk
