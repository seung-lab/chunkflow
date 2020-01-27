import os
import h5py
import numpy as np
import tifffile
import cc3d
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
    
    @classmethod
    def create(cls, size: tuple = (64, 64, 64),
               dtype: type = np.uint8, voxel_offset: tuple = (0, 0, 0)):
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)

        ix, iy, iz = np.meshgrid(*[np.linspace(0, 1, n) for 
                                   n in size[-3:]], indexing='ij')
        chunk = np.abs(np.sin(4 * (ix + iy)))
        if len(size) == 4:
            chunk = np.expand_dims(chunk, axis=0)
            chunk = np.repeat(chunk, size[0], axis=0)
            

        if dtype == np.uint8:
            chunk = (chunk * 255).astype( dtype )
        elif np.issubdtype(dtype, np.floating):
            chunk = chunk.astype(dtype)
        else:
            raise NotImplementedError()

        return cls(chunk, global_offset=voxel_offset)

    @classmethod
    def from_tif(cls, file_name: str, global_offset: tuple=None):
        arr = tifffile.imread(file_name)

        return cls(arr, global_offset=global_offset)
    
    def to_tif(self, file_name: str, global_offset: tuple=None):
        print('write chunk to file: ', file_name)
        if self.dtype==np.float32:
            # visualization in float32 is not working correctly in ImageJ
            # this might not work correctly if you want to save the image as it is!
            img = self*255 
            img = img.astype( np.uint8 )
        else:
            img = self
        tifffile.imwrite(file_name, data=img)

    @classmethod
    def from_h5(cls, file_name: str,
                dataset_path: str = '/main',
                global_offset: tuple = None):

        assert os.path.exists(file_name)
        assert h5py.is_hdf5(file_name)

        print('read from HDF5 file: {}'.format(file_name))

        global_offset_path = os.path.join(os.path.dirname(file_name),
                                       'global_offset')
        with h5py.File(file_name, 'r') as f:
            arr = np.asarray(f[dataset_path])

            if global_offset is None:
                if global_offset_path in f:
                    global_offset = tuple(f[global_offset_path])

        print('global offset: {}'.format(global_offset))

        return cls(arr, global_offset=global_offset)

    def to_h5(self, file_name: str):
        assert '.h5' in file_name

        print('write chunk to file: ', file_name)
        if os.path.exists(file_name):
            os.remove(file_name)

        with h5py.File(file_name, 'w') as f:
            f.create_dataset('/main', data=self)
            f.create_dataset('/global_offset', data=self.global_offset)

    @property
    def slices(self) -> tuple:
        """
        :getter: the global slice in the big volume
        """
        return tuple(
            slice(o, o + s) for o, s in zip(self.global_offset, self.shape))

    @property
    def is_image(self) -> bool:
        return self.ndim == 3 and self.dtype == np.uint8

    @property 
    def is_segmentation(self) -> bool:
        return self.ndim == 3 and np.issubdtype(self.dtype, np.integer) and self.dtype != np.uint8

    @property
    def is_affinity_map(self) -> bool:
        return self.ndim == 4 and np.dtype == np.float32

    @property
    def bbox(self) -> Bbox:
        """
        :getter: the cloudvolume bounding box in the big volume
        """
        return Bbox.from_delta(self.global_offset, self.shape)

    def squeeze_channel(self) -> np.ndarray:
        """given a 4D array, squeeze the channel axis."""
        assert self.ndim == 4
        axis = 0
        arr = np.squeeze(self, axis=0)
        global_offset = self.global_offset[:axis] + self.global_offset[axis+1:]
        return Chunk(arr, global_offset=global_offset)
    
    def crop_margin(self, margin_size: tuple = None, output_bbox: Bbox=None):

        if margin_size:
            if len(margin_size) == 3 and self.ndim == 4:
                margin_size = (0,) + margin_size

            if self.ndim == 3:
                chunk = self[margin_size[0]:self.shape[0] - margin_size[0],
                             margin_size[1]:self.shape[1] - margin_size[1],
                             margin_size[2]:self.shape[2] - margin_size[2]]
            elif self.ndim == 4:
                chunk = self[margin_size[0]:self.shape[0] - margin_size[0],
                             margin_size[1]:self.shape[1] - margin_size[1],
                             margin_size[2]:self.shape[2] - margin_size[2],
                             margin_size[3]:self.shape[3] - margin_size[3]]
            else:
                raise ValueError('the array dimension can only by 3 or 4.')
            global_offset = tuple(
                o + m for o, m in zip(chunk.global_offset, margin_size))
            return Chunk(chunk, global_offset=global_offset)
        else:
            print('automatically crop the chunk to output bounding box.')
            return self.cutout(output_bbox.to_slices())
    
    def connected_component(self, threshold: float = 0.5, 
                            connectivity: int = 26):
        """threshold the map chunk and get connected components."""
        global_offset = self.global_offset
        seg = self > threshold
        seg = cc3d.connected_components(seg, connectivity=connectivity)
        return Chunk(seg, global_offset=global_offset)

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
        internal_slices = tuple(
            slice(max(s.start - o, 0), min(s.stop - o, h)) for s, o, h in 
            zip(patch.slices, self.global_offset, self.shape)
        )
        shape = (s.stop - s.start for s in internal_slices)
        patch_starts = (
            i.start - s.start + o for s, o, i in 
            zip(patch.slices, self.global_offset, internal_slices)
        )
        patch_slices = tuple(slice(s, s+h) for s, h in zip(patch_starts, shape))

        self[internal_slices] += patch[patch_slices]

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
