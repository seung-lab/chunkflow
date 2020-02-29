import numpy as np

from chunkflow.chunk import Chunk
from .base import OperatorBase

from waterz import agglomerate


class AgglomerateOperator(OperatorBase):
    """Mean/max agglomeration of affinity map including watershed step."""
    def __init__(self, verbose: bool = True, name: str = 'agglomerate',
                 threshold: float = 0.7,
                 aff_threshold_low: float = 0.0001,
                 aff_threshold_high: float = 0.9999,
                 scoring_function: str = 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
                 flip_channel: bool = True):
        super().__init__(name=name, verbose=verbose)
        self.threshold = threshold
        self.aff_threshold_low = aff_threshold_low
        self.aff_threshold_high = aff_threshold_high
        self.scoring_function = scoring_function
        self.flip_channel = flip_channel

    def __call__(self, affs: np.ndarray, fragments: np.ndarray = None):
        """
        Parameters:
        -----------
        affs: affinity map with 4 dimensions: channel, z, y, x
        """
        if isinstance(affs, Chunk):
            # the segmentation is 3d, so we only need the zyx
            global_offset = affs.global_offset[-3:]
        else:
            global_offset = None

        # our affinity map channel order is x,y,z!
        # meaning the first channel is x, the second is y,
        # the third is z. We have to reverse the channel.
        if self.flip_channel:
            affs = np.flip(affs, axis=0)
        
        # waterz need datatype float32 and 
        # the array memory layout is contiguous
        affs = np.ascontiguousarray(affs, dtype=np.float32)

        # the output is a generator, and the computation is delayed
        seg_generator = agglomerate(
            affs, [self.threshold], fragments=fragments,
            aff_threshold_low=self.aff_threshold_low,
            aff_threshold_high=self.aff_threshold_high,
            scoring_function=self.scoring_function,
            force_rebuild=False)
        # there is only one threshold, so there is also only one result
        # in generator
        seg = next(seg_generator)
        return Chunk(seg, global_offset=global_offset)
