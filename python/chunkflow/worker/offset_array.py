import numpy as np
# from patch_mask import PatchMask
# from typing import Tuple
# Offset = Tuple[int, int, int]


class OffsetArray(object):
    """
        OffsetArray
    a chunk of big array with offset
    this is a python alternative of Julia OffsetArrays.jl
    https://github.com/JuliaArrays/OffsetArrays.jl
    """
    def __init__(self, array, offset):
        isinstance(array, np.ndarray)
        self.array = array
        self.offset = offset
        assert isinstance(array, np.ndarray)

    @property
    def indices(self):
        return map(lambda o, s: range(o, o+s), self.offset, self.shape)

    @property
    def parent(self):
        return self.array

    def normalize(self, mask):
        self.array *= mask

    def __getitem__(self, slices):
        internalSlices = self._get_internal_slices(self, slices)
        return self.array[internalSlices]

    def __setitem__(self, slices, inputArr):
        internalSlices = self._get_internal_slices(self, slices)
        self.array[internalSlices] = inputArr

    def __iadd__(self, other):
        indices1 = self.indices
        indices2 = other.indices
        overlap_slices = (slice(max(i1.start, i2.start), min(i1.stop, i2.stop))
                          for i1 in indices1 for i2 in indices2)
        self[overlap_slices] += other[overlap_slices]

    def __itruediv__(self, otherArray):
        # assert isinstance(otherArray, np.ndarray)
        self.array /= otherArray

    def _get_internal_slices(self, slices):
        return (slice(s.start-o, s.stop-o) for s in slices
                for o in self.offset)
