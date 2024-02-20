import numpy as np
from chunkflow.chunk import Chunk

from scipy.ndimage import gaussian_filter


def execute(chunk: Chunk, sigma: float=1.):
    filtered = chunk.clone()
    for z in range(chunk.shape[-3]):
        if chunk.ndim == 4:
            for channel in range(chunk.shape[0]):
                filtered.array[channel, z,:,:] = gaussian_filter(
                    chunk.array[channel,z,:,:], sigma=sigma)
        elif chunk.ndim == 3:
            filtered.array[z,:,:] = gaussian_filter(
                chunk.array[z,:,:], sigma=sigma)
        else:
            raise ValueError(f'only support 4 or 3d, but got {chunk.ndim}')

    # 2d processing in xy plane
    # filtered.array = gaussian_filter(chunk.array, sigma=sigma)
    return filtered
