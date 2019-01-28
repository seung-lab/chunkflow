import numpy as np

# from typing import Tuple
# Offset = Tuple[int, int, int]


class OffsetArray(np.ndarray):
    """
        OffsetArray
    a chunk of big array with offset
    this is a python alternative of Julia OffsetArrays.jl
    https://github.com/JuliaArrays/OffsetArrays.jl
    implementation following an example in (ndarray subclassing)
    [https://docs.scipy.org/doc/numpy/user/basics.subclassing.html]
    """

    def __new__(cls, array, global_offset=(0, 0, 0)):
        isinstance(array, np.ndarray)
        assert array.ndim == len(global_offset)
        obj = np.asarray(array).view(cls)
        obj.global_offset = global_offset
        return obj

    @classmethod
    def from_bbox(cls, array, bbox):
        global_offset = (bbox.minpt.z, bbox.minpt.y, bbox.minpt.x)
        return OffsetArray(array, global_offset=global_offset)

    def __array_finalize__(self, obj):
        if obj is not None:
            self.info = getattr(obj, 'global_offset', None)

    @property
    def slices(self):
        return tuple(
            slice(o, o + s) for o, s in zip(self.global_offset, self.shape))

    def where(self, mask):
        """
        find the indexes of masked value as an alternative of np.where function
        args:
            mask (binary ndarray):
        """
        isinstance(mask, np.ndarray)
        assert mask.shape == self.shape
        return tuple(i + o for i, o in zip(np.where(mask), self.global_offset))

    def add_overlap(self, other):
        assert isinstance(other, OffsetArray)
        overlap_slices = self._get_overlap_slices(other.slices)
        self[overlap_slices] += other[overlap_slices]

    def cutout(self, slices):
        if len(slices) == self.ndim - 1:
            slices = (slice(0, self.shape[0]), ) + slices
        internalSlices = self._get_internal_slices(slices)
        arr = self[internalSlices]
        global_offset = tuple(s.start for s in slices)
        return OffsetArray(arr, global_offset=global_offset)

    def save(self, patch):
        internalSlices = self._get_internal_slices(patch.slices)
        self[internalSlices] = patch

    def blend(self, patch):
        internalSlices = self._get_internal_slices(patch.slices)
        self[internalSlices] += patch

    def _get_overlap_slices(self, other_slices):
        return tuple(
            slice(max(s1.start, s2.start), min(s1.stop, s2.stop))
            for s1, s2 in zip(self.slices, other_slices))

    def _get_internal_slices(self, slices):
        return tuple(
            slice(s.start - o, s.stop - o)
            for s, o in zip(slices, self.global_offset))
