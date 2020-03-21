from typing import Union
import os
from numbers import Number
import h5py
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

import tifffile
import cc3d
from cloudvolume.lib import Bbox, yellow
# from typing import Tuple
# Offset = Tuple[int, int, int]
from .validate import validate_by_template_matching


class Chunk(NDArrayOperatorsMixin):
    r"""
       Chunk 
    
    a chunk of big array with offset
    implementation using numpy `dispatch<
    https://docs.scipy.org/doc/numpy/user/basics.dispatch.html#module-numpy.doc.dispatch>`_.
    and `examples<https://docs.scipy.org/doc/numpy/user/basics.dispatch.html#module-numpy.doc.dispatch>`_.

    :param array: the data array chunk in a big dataset
    :param global_offset: the offset of this array chunk
    :return: a new chunk with array data and global offset
    """
    def __init__(self, array, global_offset: tuple = None):
        assert isinstance(array, np.ndarray) or isinstance(array, Chunk)
        self.array = array
        if global_offset is None:
            if isinstance(array, Chunk):
                self.array = array.array
                global_offset = array.global_offset
            else:
                global_offset = tuple(np.zeros(array.ndim, dtype=np.int))
        self.global_offset = global_offset
        assert array.ndim == len(global_offset)
        
    # One might also consider adding the built-in list type to this
    # list, to support operations like np.add(array_like, list)
    _HANDLED_TYPES = (np.ndarray, Number)
    
    @classmethod
    def from_bbox(cls, array: np.ndarray, bbox: Bbox):
        """
        :param array: ndarray data
        :param bbox: cloudvolume bounding box 
        :return: construct a new Chunk
        """
        global_offset = (bbox.minpt.z, bbox.minpt.y, bbox.minpt.x)
        return cls(array, global_offset=global_offset)
    
    @classmethod
    def from_bbox(cls, bbox: Bbox, dtype: type = np.uint8):
        assert isinstance(bbox, Bbox)
        size = bbox.maxpt - bbox.minpt
        return cls.create(size=size, dtype=dtype, voxel_offset=bbox.minpt)

    @classmethod
    def create(cls, size: tuple = (64, 64, 64),
               dtype: type = np.uint8, voxel_offset: tuple = (0, 0, 0),
               all_zero: bool = False):
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)

        if all_zero:
            chunk = np.zeros(size, dtype=dtype)
        else:
            ix, iy, iz = np.meshgrid(*[np.linspace(0, 1, n) for 
                                       n in size[-3:]], indexing='ij')
            chunk = np.abs(np.sin(4 * (ix + iy)))
            if len(size) == 4:
                chunk = np.expand_dims(chunk, axis=0)
                chunk = np.repeat(chunk, size[0], axis=0)

            if np.dtype(dtype) == np.uint8:
                chunk = (chunk * 255).astype( dtype )
            elif np.dtype(dtype) == np.uint32:
                chunk = (chunk>0.5).astype(dtype) 
            elif np.issubdtype(dtype, np.floating):
                chunk = chunk.astype(dtype)
            else:
                raise NotImplementedError()

        return cls(chunk, global_offset=voxel_offset)

    @classmethod
    def from_tif(cls, file_name: str, global_offset: tuple=None):
        arr = tifffile.imread(file_name)

        return cls(arr, global_offset=global_offset)
    
    def to_tif(self, file_name: str=None, global_offset: tuple=None):
        if file_name is None:
            file_name = f'{self.bbox.to_filename()}.tif'
        print('write chunk to file: ', file_name)

        if self.array.dtype==np.float32:
            # visualization in float32 is not working correctly in ImageJ
            # this might not work correctly if you want to save the image as it is!
            print(yellow('transforming data type from float32 to uint8'))
            img = self.array*255 
            img = img.astype( np.uint8 )
        else:
            img = self.array
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
            f.create_dataset('/main', data=self.array)
            f.create_dataset('/global_offset', data=self.global_offset)
    
    def __array__(self):
        return self.array
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        example reference: 
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html?highlight=__array_ufunc__
        """
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (Chunk,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.array if isinstance(x, Chunk) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.array if isinstance(x, Chunk) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)
        
        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x, global_offset=self.global_offset) for x in result)
        elif method == 'at':
            # no return value
            return None
        elif isinstance(result, Number):
            return result
        elif isinstance(result, np.ndarray):
            # one return value
            return type(self)(result, global_offset=self.global_offset)
        else:
            return result

    def __getitem__(self, index):
        return self.array[index]
    
    def __setitem__(self, key, value):
        self.array[key] = value

    def __repr__(self):
        return f'array: {self.array}\n global offset: {self.global_offset}'
    
    def __eq__(self, value):
        if isinstance(value, type(self)):
            return np.array_equal(self.array, value.array) and np.array_equal(
                self.global_offset, value.global_offset)
        elif isinstance(value, Number):
            return np.all(self.array==value)
        elif isinstance(value, np.ndarray):
            return np.all(self.array == value)
        else:
            raise NotImplementedError

    @property
    def slices(self) -> tuple:
        """
        :getter: the global slice in the big volume
        """
        return tuple(
            slice(o, o + s) for o, s in zip(self.global_offset, self.shape))

    @property
    def is_image(self) -> bool:
        return self.array.ndim == 3 and self.array.dtype == np.uint8

    @property 
    def is_segmentation(self) -> bool:
        return self.array.ndim == 3 and np.issubdtype(
            self.array.dtype, np.integer) and self.array.dtype != np.uint8

    @property
    def is_affinity_map(self) -> bool:
        return self.array.ndim == 4 and self.array.dtype == np.float32

    @property
    def bbox(self) -> Bbox:
        """
        :getter: the cloudvolume bounding box in the big volume
        """
        return Bbox.from_delta(self.global_offset, self.array.shape)
    
    @property
    def ndim(self) -> int:
        return self.array.ndim 

    @property 
    def shape(self) -> tuple:
        return self.array.shape 
    
    @property 
    def dtype(self) -> np.dtype:
        return self.array.dtype 
    
    def astype(self, dtype: np.dtype):
        if dtype != self.array.dtype:
            new_array = self.array.astype(dtype)
            return type(self)(new_array, global_offset=self.global_offset)
        else:
            return self
    
    def max(self, *args, **kwargs):
        return self.array.max(args, kwargs)

    def min(self, *args, **kwargs):
        return self.array.min(args, kwargs)

    def transpose(self):
        """To-Do: support arbitrary axis transpose"""
        new_array = self.array.transpose()
        new_global_offset = self.global_offset[::-1]
        return type(self)(new_array, global_offset=new_global_offset)
    
    def fill(self, x):
        self.array.fill(x)

    def squeeze_channel(self) -> np.ndarray:
        """given a 4D array, squeeze the channel axis."""
        assert self.array.ndim == 4
        axis = 0
        arr = np.squeeze(self, axis=0)
        global_offset = self.global_offset[:axis] + self.global_offset[axis+1:]
        return Chunk(arr, global_offset=global_offset)

    def channel_voting(self):
        assert self.ndim == 4
        assert self.shape[0] <= 256
        out = np.empty(self.shape[1:], dtype=np.uint8)
        np.argmax(self.array, axis=0, out=out)
        # our selected channel index start from 1
        out += 1
        return Chunk(out, global_offset=self.global_offset[1:])

    def mask_using_last_channel(self, threshold: float = 0.3) -> np.ndarray:
        assert self.ndim==4

        mask = (self.array[-1, :, :, :] < threshold)
        ret = self.array[:-1, ...]
        ret *= mask
        return Chunk(ret, global_offset=self.global_offset)

    def crop_margin(self, margin_size: tuple = None, output_bbox: Bbox=None):

        if margin_size:
            if len(margin_size) == 3 and self.ndim == 4:
                margin_size = (0,) + margin_size

            if self.ndim == 3:
                new_array = self.array[margin_size[0]: -margin_size[0],
                                       margin_size[1]: -margin_size[1],
                                       margin_size[2]: -margin_size[2]]
            elif self.ndim == 4:
                new_array = self.array[margin_size[0]: -margin_size[0],
                                       margin_size[1]: -margin_size[1],
                                       margin_size[2]: -margin_size[2],
                                       margin_size[3]: -margin_size[3]]
            else:
                raise ValueError('the array dimension can only by 3 or 4.')
            global_offset = tuple(
                o + m for o, m in zip(self.global_offset, margin_size))
            return Chunk(new_array, global_offset=global_offset)
        else:
            print('automatically crop the chunk to output bounding box.')
            assert output_bbox is not None
            return self.cutout(output_bbox.to_slices())
    
    def connected_component(self, threshold: float = 0.5, 
                            connectivity: int = 26):
        """threshold the map chunk and get connected components."""
        global_offset = self.global_offset
        seg = self > threshold
        seg = cc3d.connected_components(seg.array, connectivity=connectivity)
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
        self.array[overlap_slices] += other.array[overlap_slices]

    def cutout(self, slices: tuple):
        """
        cutout a region of interest from this chunk

        :param slices: the global slices of region of interest
        :return: another chunk of region of interest
        """
        if len(slices) == self.ndim - 1:
            slices = (slice(0, self.shape[0]), ) + slices
        internalSlices = self._get_internal_slices(slices)
        arr = self.array[internalSlices]
        global_offset = tuple(s.start for s in slices)
        return Chunk(arr, global_offset=global_offset)

    def save(self, patch):
        """
        replace a region of interest from another chunk

        :param patch: a small chunk to replace subvolume
        """
        internalSlices = self._get_internal_slices(patch.slices)
        self.array[internalSlices] = patch.array

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

        self.array[internal_slices] += patch.array[patch_slices]

    def _get_overlap_slices(self, other_slices):
        return tuple(
            slice(max(s1.start, s2.start), min(s1.stop, s2.stop))
            for s1, s2 in zip(self.slices, other_slices))

    def _get_internal_slices(self, slices):
        return tuple(
            slice(s.start - o, s.stop - o)
            for s, o in zip(slices, self.global_offset))


    def validate(self, verbose: bool = False):
        """validate the completeness of this chunk, there
        should not have black boxes.

        :param verbose: show detailed info or not
        """
        validate_by_template_matching(self.array, verbose=verbose)

