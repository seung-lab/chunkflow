import numpy as np

from chunkflow.chunk import Chunk
from chunkflow.chunk.probability_map import ProbabilityMap
from chunkflow.synapses import Synapses
from chunkflow.lib.cartesian_coordinate import BoundingBox


def execute(prob: Chunk, bbox: BoundingBox):
    if prob is None:
        print('get None probability map!')
        return None

    if np.issubdtype(prob.dtype, np.uint8):
        prob = prob.astype(np.float32)
        prob /= 255.
    prob = ProbabilityMap.from_chunk(prob)

    pre, pre_confidences = prob.detect_points(min_distance=15, threshold_rel=0.3)
    if pre is None:
        syns = None
    else:
        syns = Synapses(pre, pre_confidence=pre_confidences, resolution=prob.voxel_size)
        syns.remove_synapses_outside_bounding_box(bbox)

    return syns