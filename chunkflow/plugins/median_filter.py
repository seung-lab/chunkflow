
import numpy as np
from scipy.ndimage import median_filter

def exec(chunk: np.ndarray, size: tuple=(3,1,1), mode: str='reflect'):
    return median_filter(chunk, size=size, mode=mode)

    