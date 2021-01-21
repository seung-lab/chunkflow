import numpy as np

def execute(chunk: np.ndarray):
    if np.issubdtype(chunk.dtype, np.uint8):
        chunk = 255 - chunk
    elif np.issubdtype(chunk.dtype, np.float32) and chunk.max() <= 1 and chunk.min()>=0 :
        chunk = 1.0 - chunk
    else:
        raise TypeError("unsupported chunk data type.")
    return [chunk]
