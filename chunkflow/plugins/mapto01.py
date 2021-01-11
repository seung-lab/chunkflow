import numpy as np

def execute(chunk: np.ndarray):
    if np.issubdtype(chunk.dtype, np.uint8):
        chunk = chunk.astype(np.float32)
        chunk = chunk / 255.0
    assert np.issubdtype(chunk.dtype, np.float32)
    assert chunk.min() >= -1
    assert chunk.max() <= 1
    chunk = (chunk + 1.0) / 2.0
    return [chunk]
