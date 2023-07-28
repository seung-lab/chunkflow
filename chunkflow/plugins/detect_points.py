import numpy as np

from chunkflow.chunk import Chunk
from chunkflow.chunk.probability_map import ProbabilityMap
from chunkflow.point_cloud import PointCloud


def execute(prob: Chunk,
        min_distance: float = 15.,
        threshold_rel: float = 0.3):
    if prob is None:
        print('get None probability map!')
        return None
    
    assert threshold_rel > 0.
    assert threshold_rel < 1.

    if np.issubdtype(prob.dtype, np.uint8):
        prob = prob.astype(np.float32)
        prob /= 255.
    prob = ProbabilityMap.from_chunk(prob)

    # drop confidence for now
    points, _= prob.detect_points(
        min_distance=min_distance, 
        threshold_rel=threshold_rel
    )

    points = PointCloud(points, prob.voxel_size)
    return points