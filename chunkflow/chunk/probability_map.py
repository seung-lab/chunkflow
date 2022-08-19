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
            exclude_border: int=True, indices: bool=True, 
            num_peaks_per_label: int = math.inf, return_confidences: bool = True):
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
            indices=indices,
            num_peaks_per_label = num_peaks_per_label,
        )
        print('number of detected points: ', coords.shape[0])
        if len(coords) == 0:
            coords = None
            if return_confidences:
                confidences = None
        else:
            if return_confidences:
                confidences = prob[coords[:, 0], coords[:, 1], coords[:, 2]]
            coords += np.asarray(self.voxel_offset, dtype=coords.dtype).reshape(1,3)

        if return_confidences:
            return coords, confidences
        else:
            return coords