import numpy as np
from cloudvolume.lib import Bbox
# from typing import Tuple
# Offset = Tuple[int, int, int]
from .validate import validate_by_template_matching


class Chunk(np.ndarray):
    r"""
       Chunk 
    
    a chunk of big array with offset
    implementation following an example in ndarray `subclassing
    <https://docs.scipy.org/doc/numpy/user/basics.subclassing.html>`_.

    :param array: the data array chunk in a big dataset
    :param global_offset: the offset of this array chunk
    :return: a new chunk with array data and global offset
    """

    def __new__(cls, array: np.ndarray, global_offset: tuple = None):
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
                obj, 'global_offset', tuple(np.zeros(obj.ndim, dtype=np.int)))

    @classmethod
    def from_bbox(cls, array: np.ndarray, bbox: Bbox):
        """
        :param array: ndarray data
        :param bbox: cloudvolume bounding box 
        :return: construct a new Chunk
        """
        global_offset = (bbox.minpt.z, bbox.minpt.y, bbox.minpt.x)
        return Chunk(array, global_offset=global_offset)

    @property
    def slices(self) -> tuple:
        """
        :getter: the global slice in the big volume
        """
        return tuple(
            slice(o, o + s) for o, s in zip(self.global_offset, self.shape))

    @property
    def bbox(self) -> Bbox:
        """
        :getter: the cloudvolume bounding box in the big volume
        """
        return Bbox.from_delta(self.global_offset, self.shape)

    def where(self, mask: np.ndarray) -> tuple:
        """
        find the indexes of masked value as an alternative of np.where function

        :param mask: binary ndarray as mask
        :return: the coordinates in global coordinates.
        """
        isinstance(mask, np.ndarray)
        assert mask.shape == self.shape
        return tuple(i + o for i, o in zip(np.where(mask), self.global_offset))

    def add_overlap(self, other):
        """
        sum up overlaping region with another chunk

        :param other: another chunk
        :return: sum up result.
        """
        assert isinstance(other, Chunk)
        overlap_slices = self._get_overlap_slices(other.slices)
        self[overlap_slices] += other[overlap_slices]

    def cutout(self, slices: tuple):
        """
        cutout a region of interest from this chunk

        :param slices: the global slices of region of interest
        :return: another chunk of region of interest
        """
        if len(slices) == self.ndim - 1:
            slices = (slice(0, self.shape[0]), ) + slices
        internalSlices = self._get_internal_slices(slices)
        arr = self[internalSlices]
        global_offset = tuple(s.start for s in slices)
        return Chunk(arr, global_offset=global_offset)

    def save(self, patch):
        """
        replace a region of interest from another chunk

        :param patch: a small chunk to replace subvolume
        """
        internalSlices = self._get_internal_slices(patch.slices)
        self[internalSlices] = patch

    def blend(self, patch):
        """
        same with add_overlap
        """
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

    def validate(self, verbose: bool = False):
        """validate the completeness of this chunk, there
        should not have black boxes.

        :param verbose: show detailed info or not
        """
        validate_by_template_matching(self, verbose=verbose)
