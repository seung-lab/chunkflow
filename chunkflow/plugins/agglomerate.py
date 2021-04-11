import numpy as np

from chunkflow.chunk import Chunk

from waterz import agglomerate


threshold: float = 0.7
aff_threshold_low: float = 0.001
aff_threshold_high: float = 0.9999
scoring_function: str = 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>'
flip_channel: bool = True


def execute(affs: Chunk, fragments: np.ndarray = None):
    """
    Mean/max agglomeration of affinity map including watershed step.

    Parameters:
    -----------
    affs: affinity map with 4 dimensions: channel, z, y, x
    """
    properties = affs.properties

    # our affinity map channel order is x,y,z!
    # meaning the first channel is x, the second is y,
    # the third is z. We have to reverse the channel for waterz.
    if flip_channel:
        affs = np.flip(affs, axis=0)
    
    # waterz need datatype float32 and 
    # the array memory layout is contiguous
    affs = np.ascontiguousarray(affs, dtype=np.float32)

    # the output is a generator, and the computation is delayed
    seg_generator = agglomerate(
        affs, [threshold], fragments=fragments,
        aff_threshold_low=aff_threshold_low,
        aff_threshold_high=aff_threshold_high,
        scoring_function=scoring_function,
        force_rebuild=False)
    # there is only one threshold, so there is also only one result
    # in generator
    seg = next(seg_generator)
    seg = Chunk(seg)
    seg.set_properties(properties)

    return [seg]
