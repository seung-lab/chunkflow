import numpy as np
from cloudvolume.lib import Bbox
# from typing import Tuple
# Offset = Tuple[int, int, int]


class Chunk(np.ndarray):
    """
       Chunk 
    a chunk of big array with offset
    implementation following an example in (ndarray subclassing)
    [https://docs.scipy.org/doc/numpy/user/basics.subclassing.html]
    """

    def __new__(cls, array, global_offset=None):
        if global_offset is None:
            global_offset = tuple(np.zeros(array.ndim, dtype=np.int))
        assert array.ndim == len(global_offset)
        obj = np.asarray(array).view(cls)
        obj.global_offset = global_offset
        return obj
    
    def __array_finalize__(self, obj):
        """
        https://www.numpy.org/devdocs/user/basics.subclassing.html#basics-subclassing
        """
        if obj is None:
            return
        else:
            self.global_offset = getattr(
                obj, 'global_offset',
                tuple(np.zeros(obj.ndim, dtype=np.int))
            )

    @classmethod
    def from_bbox(cls, array, bbox):
        global_offset = (bbox.minpt.z, bbox.minpt.y, bbox.minpt.x)
        return Chunk(array, global_offset=global_offset)

    @property
    def slices(self):
        return tuple(slice(o, o + s) for o, s in 
                     zip(self.global_offset, self.shape))

    @property
    def bbox(self):
        return Bbox.from_delta(self.global_offset, self.shape)

    def where(self, mask):
        """
        find the indexes of masked value as an alternative of np.where function
        args:
            mask (binary ndarray):
        """
        isinstance(mask, np.ndarray)
        assert mask.shape == self.shape
        return tuple(i + o for i, o in 
                     zip(np.where(mask), self.global_offset))

    def add_overlap(self, other):
        assert isinstance(other, Chunk)
        overlap_slices = self._get_overlap_slices(other.slices)
        self[overlap_slices] += other[overlap_slices]

    def cutout(self, slices):
        if len(slices) == self.ndim - 1:
            slices = (slice(0, self.shape[0]), ) + slices
        internalSlices = self._get_internal_slices(slices)
        arr = self[internalSlices]
        global_offset = tuple(s.start for s in slices)
        return Chunk(arr, global_offset=global_offset)

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

#    def __array_wrap__(self, out_arr, context=None):
#        chunk = super().__array_wrap__(self, out_arr, context)
#        return Chunk(chunk, global_offset=self.global_offset)
