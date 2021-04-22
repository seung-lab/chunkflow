
import numpy as np
from scipy.ndimage import median_filter

def execute(chunk: np.ndarray, size: tuple=(3,1,1), mode: str='reflect'):
    print('median filtering of chunk...')
    chunk = median_filter(chunk, size=size, mode=mode)
    return [chunk]
