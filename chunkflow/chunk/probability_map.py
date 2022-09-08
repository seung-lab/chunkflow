import math

import numpy as np
from skimage.feature import peak_local_max

from chunkflow.lib.cartesian_coordinate import Cartesian
from .base import Chunk


class ProbabilityMap(Chunk):
    def __init__(self, array: np.ndarray, 
            voxel_offset: Cartesian = None, 
            voxel_size: Cartesian = None):
        super().__init__(array, voxel_offset, voxel_size)
        assert np.issubdtype(self.dtype, np.floating)

    @classmethod
    def from_chunk(cls, chunk: Chunk):
        return cls(chunk.array, chunk.voxel_offset, chunk.voxel_size)

    def detect_points(self, min_distance: int = 1, threshold_rel: float=0.3, 
            exclude_border: int=True, num_peaks: int = math.inf, 
            return_confidences: bool = True):
        # prob = chunk.array.copy()
        prob = self.array
        prob -= np.mean(prob)
        prob[prob<0.] = 0.
        prob /= prob.max()

        # since we have already normalized the map, 
        # the absolute threshold becomes relative threshold!
        coords = peak_local_max(
            prob, 
            min_distance=min_distance, 
            threshold_abs=threshold_rel,
            exclude_border=exclude_border,
            num_peaks = num_peaks,
        )
        print('number of detected points: ', coords.shape[0])
        if coords.shape[0] > num_peaks:
            print(f'only select the first {num_peaks} points.')
            coords = coords[:num_peaks, :]
        
        if len(coords) == 0:
            coords = None
            if return_confidences:
                confidences = None
        else:
            if return_confidences:
                confidences = prob[coords[:, 0], coords[:, 1], coords[:, 2]]
            coords +=self.voxel_offset
       
        if return_confidences:
            return coords, confidences
        else:
            return coords
