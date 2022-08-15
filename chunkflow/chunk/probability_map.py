import math

import numpy as np
from .base import Chunk
from skimage.feature import peak_local_max

from chunkflow.lib.cartesian_coordinate import Cartesian


class ProbabilityMap(Chunk):
    def __init__(self, array: np.ndarray, voxel_offset: Cartesian = None, voxel_size: Cartesian = None):
        super().__init__(array, voxel_offset, voxel_size)
        assert np.issubdtype(self.dtype, np.floating)

    def detect_points(self, min_distance: int = 1, threshold_abs: float=None, 
            threshold_rel: float = None, exclude_border: int=True, indices: bool=True, 
            num_peaks_per_label: int = math.inf):
        # prob = chunk.array.copy()
        prob = self.array
        prob -= np.mean(prob)
        prob[prob<0.] = 0.
        prob /= prob.max()

        coords = peak_local_max(
            prob, 
            min_distance=min_distance, 
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel,
            exclude_border=exclude_border,
            indices=indices,
            num_peaks_per_label = num_peaks_per_label,
        )

        confidences = prob[coords[:, 0], coords[:, 1], coords[:, 2]]
        
        coords += np.asarray(self.voxel_offset, dtype=coords.dtype).reshape(1,3)
        print('number of detected points: ', coords.shape[0])
        return coords, confidences