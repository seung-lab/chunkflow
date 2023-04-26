from __future__ import annotations
import logging
from typing import Union, Optional
import os
from numbers import Number

import h5py
import numpy as np
import nrrd
from numpy.core.numerictypes import issubdtype
from numpy.lib.mixins import NDArrayOperatorsMixin

import tifffile
import cc3d
from cloudvolume.lib import yellow, Bbox

from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian

# from typing import Tuple
# Offset = Tuple[int, int, int]
from .validate import validate_by_template_matching

def layer_type_is_valid(type: str):
    return type in set([None, 'image', 'segmentation', 'probability_map', 'affinity_map', 'unknown'])


class Chunk(NDArrayOperatorsMixin):
    def __init__(self, array: np.ndarray, 
            voxel_offset: Cartesian = None, 
            voxel_size: Cartesian = None,
            layer_type: str = None):
        """chunk of a volume
    
        a chunk of big array with offset
        implementation using numpy `dispatch<
        https://docs.scipy.org/doc/numpy/user/basics.dispatch.html#module-numpy.doc.dispatch>`_.
        and `examples<https://docs.scipy.org/doc/numpy/user/basics.dispatch.html#module-numpy.doc.dispatch>`_.

        Args:
            array (np.ndarray): the data
            voxel_offset (Cartesian, optional): voxel offset. Defaults to None.
            voxel_size (Cartesian, optional): voxel size. Defaults to None.
            type (str, optional): type of chunk. [None, image, segmentation, probability_map, affinity_map, unknown]. Defaults to None.
        
        Return: 
            a new chunk with array data and global offset
        """
        if array.ndim == 2:
            array = np.expand_dims(array, axis=0)
        else:
            assert array.ndim >= 3 and array.ndim <= 4
        assert isinstance(array, np.ndarray) or isinstance(array, Chunk)
        assert layer_type_is_valid(layer_type), f'layer type: {layer_type} is unsupported!'

        self.array = array
        if voxel_offset is None:
            if isinstance(array, Chunk):
                self.array = array.array
                voxel_offset = array.voxel_offset
            else:
                voxel_offset = Cartesian(0, 0, 0)
        
        if voxel_offset is not None:
            if len(voxel_offset) == 4:
                assert voxel_offset[0] == 0
                voxel_offset = voxel_offset[1:]
            assert len(voxel_offset) == 3

        if not isinstance(voxel_offset, Cartesian):
            voxel_offset = Cartesian.from_collection(voxel_offset)
        self.voxel_offset = voxel_offset

        if voxel_size is not None and not isinstance(voxel_size, Cartesian):
            voxel_size = Cartesian.from_collection(voxel_size)
        self.voxel_size = voxel_size
        if voxel_size is not None:
            assert len(voxel_size) == 3
            assert np.alltrue([vs > 0 for vs in voxel_size])
        
        if layer_type is not None:
            self.layer_type = layer_type 
        else:
            # best guess
            if self.is_image:
                self.layer_type = 'image'
            elif self.is_segmentation:
                self.layer_type = 'segmentation'
            elif self.is_probability_map:
                self.layer_type = 'probability_map'
            elif self.is_affinity_map:
                self.layer_type = 'affinity_map'
            else:
                self.layer_type = 'unknown'

    # One might also consider adding the built-in list type to this
    # list, to support operations like np.add(array_like, list)
    _HANDLED_TYPES = (np.ndarray, Number)
    
    @classmethod
    def from_array(cls, array: np.ndarray, bbox: BoundingBox, 
            voxel_size: Optional[tuple] = None):
        """
        :param array: ndarray data
        :param bbox: cloudvolume bounding box
        :param voxel_size: physical size of each voxel.
        :return: construct a new Chunk
        """
        return cls(array, voxel_offset=bbox.minpt, voxel_size=voxel_size)
    
    @classmethod
    def from_bbox(cls, bbox: BoundingBox, dtype: type = np.uint8,
            pattern: str='zero',
            voxel_size: tuple=None):
        """create a chunk from bounding box

        Args:
            bbox (BoundingBox): 3D bounding box
            dtype (type, optional): data type. Defaults to np.uint8.
            pattern (str, optional): method to create empty array. [zero, random, sin]. Defaults to 'zero'.
            voxel_size (tuple, optional): physical size of a voxel. Defaults to None.

        Returns:
            _type_: _description_
        """
        assert isinstance(bbox, BoundingBox)
        size = bbox.maxpt - bbox.minpt
        return cls.create(size=size, dtype=dtype, voxel_offset=bbox.minpt,
            voxel_size=voxel_size, pattern=pattern)
    
    def connected_component(self, threshold: float = None, 
                            connectivity: int = 6):
        """threshold the map chunk and get connected components."""
        if not self.is_segmentation and threshold is not None:
            seg = self.threshold(threshold)
            seg = seg.array
        else:
            seg = self.array 
        seg = cc3d.connected_components(seg, connectivity=connectivity)
        return Chunk(seg, voxel_offset=self.voxel_offset, voxel_size=self.voxel_size)

    @classmethod
    def create(cls, size: Cartesian = Cartesian(64, 64, 64),
               dtype: type = np.uint8, 
               voxel_offset: Cartesian = Cartesian(0, 0, 0),
               voxel_size: Cartesian = None,
               pattern: str = 'sin',
               high: int = 255):
        """create a fake chunk for tests.

        Args:
            size (tuple, Cartesian, optional): chunk size or shape. Defaults to (64, 64, 64).
            dtype (type, optional): data type like numpy. Defaults to np.uint8. options: [uint8, uint16, uint32, uint64, float32]
            voxel_offset (Cartesian, optional): coordinate of starting voxel. Defaults to Cartesian(0, 0, 0).
            voxel_size (Cartesian, optional): physical size of each voxel. Defaults to None.
            pattern (str, optional): ways to create an array. ['sin', 'random', 'zero']. Defaults to 'sin'.
            high (int, optional): the high value of random integer array. Defaults to 255.

        Raises:
            NotImplementedError: not support pattern or data type was used.

        Returns:
            Chunk: the random chunk created.
        """
        # if not isinstance(size, Cartesian):
        #     size = Cartesian.from_collection(size)

        if isinstance(dtype, str):
            dtype = np.dtype(dtype)

        if pattern == 'zero':
            arr = np.zeros(size, dtype=dtype)
        elif pattern == 'sin':
            ix, iy, iz = np.meshgrid(*[np.linspace(0, 1, n) for 
                                       n in size[-3:]], indexing='ij')
            arr = np.abs(np.sin(4 * (ix + iy + iz)))
            if len(size) == 4:
                arr = np.expand_dims(arr, axis=0)
                arr = np.repeat(arr, size[0], axis=0)

            if dtype == np.uint8:
                arr = (arr * 255).astype( dtype )
            elif dtype==np.uint16 or dtype == np.uint32 or dtype == np.uint64:
                arr = (arr>0.5).astype(dtype)
                arr = cc3d.connected_components(arr, connectivity=6)
            elif np.issubdtype(dtype, np.floating):
                arr = arr.astype(dtype)
            else:
                raise NotImplementedError(f'do not support this data type: {dtype}')
        elif pattern == 'random':
            if np.issubdtype(dtype, np.floating):
                arr = np.random.rand(*size)
                arr = arr.astype(dtype)
            elif np.issubdtype(dtype, np.integer):
                arr = np.random.randint(high, size=size, dtype=dtype)
                arr = cc3d.connected_components(arr, connectivity=6)
            else:
                raise NotImplementedError(f'do not support this data type: {dtype}')
        else:
            raise NotImplementedError(f'do not support the pattern: {pattern}')

        return cls(arr, voxel_offset=voxel_offset, voxel_size=voxel_size)

    def clone(self):
        return Chunk(
            self.array.copy(), 
            voxel_offset=self.voxel_offset, 
            voxel_size=self.voxel_size
        )

    @classmethod
    def from_nrrd(cls, file_name: str, voxel_offset: tuple=None, dtype: str = None,
            voxel_size: tuple=None):
        arr, _ = nrrd.read(file_name)

        if dtype:
            arr = arr.astype(dtype)
        return cls(arr, voxel_offset=voxel_offset, voxel_size=voxel_size)
    
    def to_nrrd(self, file_name: str=None):
        if file_name is None:
            file_name = f'{self.bbox.string}.nrrd'
        elif not file_name.endswith('.nrrd'):
            file_name += f'_{self.bbox.string}.nrrd'

        logging.info(f'write chunk to file: {file_name}')
        nrrd.write(file_name, self.array)

    @classmethod
    def from_tif(cls, file_name: str, voxel_offset: tuple=None, dtype: str = None,
            voxel_size: tuple=None):
        arr = tifffile.imread(file_name)

        if dtype:
            arr = arr.astype(dtype)
        logging.info(f'read tif chunk with size of {arr.shape}, voxel offset: {voxel_offset}, voxel size: {voxel_size}')
        return cls(arr, voxel_offset=voxel_offset, voxel_size=voxel_size)
    
    def to_tif(self, file_name: str=None, compression: str = 'zlib'):
        if file_name is None:
            file_name = f'{self.bbox.string}.tif'
        logging.info(f'write chunk to file: {file_name}')

        if self.array.dtype==np.float32:
            # visualization in float32 is not working correctly in ImageJ
            # this might not work correctly if you want to save the image as it is!
            print(yellow('transforming data type from float32 to uint8'))
            img = self.array*255 
            img = img.astype( np.uint8 )
        else:
            img = self.array
        
        if self.ndim == 3:
            axes = 'ZYX'
        elif self.ndim == 4:
            axes = 'CZYX'
        metadata = {'spacing': 1, 'unit': 'nm', 'axes': axes}
        tifffile.imwrite(
            file_name, data=img, 
            volumetric = True,
            # resolution=self.voxel_size.tuple, 
            # imagej=True,
            metadata = metadata,
            compression = compression,
        )

    @classmethod
    def from_h5(cls, file_name: str,
                voxel_offset: tuple=None,
                dataset_path: str = None,
                voxel_size: tuple = None,
                cutout_start: tuple = None,
                cutout_stop: tuple = None,
                cutout_size: tuple = None,
                dtype: str = None,
                layer_type: str = None):

        file_name = os.path.expanduser(file_name)
        assert os.path.exists(file_name), f'the file do not exist: {file_name}'
        
        if cutout_start is not None and cutout_size is not None:
            cutout_stop = tuple(t+s for t, s in zip(cutout_start, cutout_size))

        if not h5py.is_hdf5(file_name):
            assert cutout_start is not None 
            assert cutout_stop is not None
            bbox = BoundingBox.from_list([*cutout_start, *cutout_stop])
            file_name += f'{bbox.string}.h5'

            if not os.path.exists(file_name) or os.path.getsize(file_name)==0:
                # fill with zero
                assert dtype is not None
                logging.info(f'{file_name} do not exist or is empty, will return None.')
                # return cls.from_bbox(bbox, dtype=dtype, voxel_size=voxel_size, all_zero=True)
                return None

        with h5py.File(file_name, 'r') as f:
            if dataset_path is None:
                for key in f.keys():
                    if 'offset' not in key and 'unique' not in key:
                        # the first name without offset inside
                        dataset_path = key
                        break
            dset = f[dataset_path]
            if voxel_offset is None: 
                if 'voxel_offset' in f:
                    voxel_offset = Cartesian(*f['voxel_offset'])
                else:
                    voxel_offset = Cartesian(0, 0, 0)

            if voxel_size is None:
                if 'voxel_size' in f:
                    voxel_size = Cartesian(*f['voxel_size'])
                else:
                    voxel_size = Cartesian(1, 1, 1)

            if layer_type is None:
                if 'layer_type' in f.attrs:
                    layer_type = f.attrs['layer_type']
                    # type = str(f['type'])
                    assert layer_type_is_valid(layer_type)
            
            if cutout_start is None:
                cutout_start = voxel_offset
            if cutout_size is None:
                cutout_size = dset.shape[-3:]
                cutout_size = Cartesian.from_collection(cutout_size)
            elif np.min(cutout_size) < 0:
                cutout_size = [x for x in cutout_size]
                for idx in range(-1, -4, -1):
                    if cutout_size[idx]<0:
                        cutout_size[idx] = dset.shape[idx]
                cutout_size = Cartesian.from_collection(cutout_size)
            if cutout_stop is None:
                cutout_stop = tuple(t+s for t, s in zip(cutout_start, cutout_size))

            for c, v in zip(cutout_start, voxel_offset):
                assert c >= v, "can only cutout after the global voxel offset."
            
            assert len(cutout_start) == 3
            assert len(cutout_stop) == 3
            dset = dset[...,
                cutout_start[0]-voxel_offset[0]:cutout_stop[0]-voxel_offset[0],
                cutout_start[1]-voxel_offset[1]:cutout_stop[1]-voxel_offset[1],
                cutout_start[2]-voxel_offset[2]:cutout_stop[2]-voxel_offset[2],
            ]
                    
        
        logging.info(f"""read from HDF5 file: {file_name} and start with {cutout_start}, \
ends with {cutout_stop}, size is {cutout_size}, voxel size is {voxel_size}.""")
        arr = np.asarray(dset)
        if arr.dtype == np.dtype('<f4'):
            arr = arr.astype('float32')
        elif arr.dtype == np.dtype('<f8'):
            arr = arr.astype('float64') 

        logging.info(f'new chunk voxel offset: {cutout_start}')

        return cls(arr, voxel_offset=cutout_start, voxel_size=voxel_size, layer_type=layer_type)

    def to_h5(self, file_name: str, with_offset: bool=True, 
                chunk_size: Union[Cartesian, tuple] = (8,8,8),
                with_unique: bool= True, 
                compression="gzip",
                voxel_size: tuple = None):
        """
        :param file_name: output file name. If it is not end with h5, the coordinate will be appended to the file name.
        :param with_offset: save the voxel offset or not
        :param with_unique: if this is a segmentation chunk, save the unique object ids or not.
        :param compression: use HDF5 compression or not. Options are gzip, lzf
        """
        if chunk_size:
            assert len(chunk_size) == 3
        if isinstance(chunk_size, Cartesian):
            chunk_size = tuple(*chunk_size)

        if not file_name.endswith('.h5'):
            file_name += self.bbox.string + '.h5'

        logging.info(f'write chunk to file: {file_name}')
        if os.path.exists(file_name):
            print(yellow(f'deleting existing file: {file_name}'))
            os.remove(file_name)

        with h5py.File(file_name, 'w') as f:
            f.create_dataset('/main', data=self.array, chunks=chunk_size, compression=compression)
            if voxel_size is None and self.voxel_size is not None:
                voxel_size = self.voxel_size
            if voxel_size is not None:
                f.create_dataset('/voxel_size', data=voxel_size)
            if self.layer_type is not None:
                f.attrs['layer_type'] = self.layer_type

            if with_offset and self.voxel_offset is not None:
                f.create_dataset('/voxel_offset', data=self.voxel_offset)

            if with_unique and self.is_segmentation:
                unique = np.unique(self.array)
                if unique[0]:
                    unique = unique[1:]
                f.create_dataset('/unique_nonzeros', data = unique)
        return file_name

    def __len__(self):
        return len(self.array)

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
            return tuple(type(self)(x, voxel_offset=self.voxel_offset, voxel_size=self.voxel_size) for x in result)
        elif method == 'at':
            # no return value
            return None
        elif isinstance(result, Number):
            return result
        elif isinstance(result, np.ndarray):
            # one return value
            return Chunk(result, voxel_offset=self.voxel_offset, voxel_size=self.voxel_size)
        else:
            return result

    def __getitem__(self, index):
        return self.array[index]
    
    def __setitem__(self, key, value):
        self.array[key] = value

    def __repr__(self):
        return f'array: {self.array}\n voxel offset: {self.voxel_offset} \n voxel size: {self.voxel_size}'
    
    def __eq__(self, value):
        if isinstance(value, type(self)):
            return np.array_equal(self.array, value.array) and np.array_equal(
                self.voxel_offset, value.voxel_offset)
        elif isinstance(value, Number):
            # return np.all(self.array==value)
            return self.array == value
        elif isinstance(value, np.ndarray):
            # return np.all(self.array == value)
            return self.array == value
        else:
            raise NotImplementedError


    # @property
    # def voxel_offset(self) -> Cartesian:
    #     return self.voxel_offset

    # @voxel_offset.setter
    # def voxel_offset(self, value: Cartesian):
    #     self.voxel_offset = value

    # @property
    # def voxel_size(self) -> Cartesian:
    #     return self.voxel_size

    # @voxel_size.setter
    # def voxel_size(self, value: Cartesian):
    #     self.voxel_size = value

    @property
    def is_image(self) -> bool:
        return issubdtype(self.dtype, np.uint8) and \
                    self.ndim == 3

    @property 
    def is_segmentation(self) -> bool:
        return self.array.ndim == 3 and \
                    (np.issubdtype(self.array.dtype, np.integer) or \
                        np.issubdtype(self.dtype, bool)) and \
                            self.array.dtype != np.uint8

    @property
    def is_affinity_map(self) -> bool:
        return self.array.ndim == 4 and self.shape[0] == 3 and self.array.dtype == np.float32
    
    @property
    def is_probability_map(self) -> bool:
        return self.array.ndim == 4 and self.array.dtype == np.float32

    @property
    def properties(self) -> dict:
        props = dict()
        if self.voxel_offset is not None or self.voxel_offset != Cartesian(0, 0, 0):
            props['voxel_offset'] = self.voxel_offset
        if self.voxel_size is not None or self.voxel_size != Cartesian(1, 1, 1):
            props['voxel_size'] = self.voxel_size
        if self.layer_type is not None:
            props['layer_type'] = self.layer_type

        return props 
    
    def set_properties(self, properties: dict):
        if 'voxel_offset' in properties:
            self.voxel_offset = properties['voxel_offset']

        if 'voxel_size' in properties:
            self.voxel_size = properties['voxel_size']
        
        if 'layer_type' in properties:
            self.layer_type = properties['layer_type']

    @properties.setter
    def properties(self, value: dict):
        self.set_properties(value)

    @property
    def flags(self):
        return self.array.flags

    @property
    def size(self):
        return self.array.size

    @property
    def slices(self) -> tuple:
        """
        :getter: the global slice in the big volume
        """
        return tuple(
            slice(o, o + s) for o, s in zip(self.ndoffset, self.shape))
    
    @property
    def ndoffset(self) -> tuple:
        """ 
        make the voxel offset have the same dimension with array
        """
        if self.ndim == 4:
            return (0, *self.voxel_offset)
        else:
            return self.voxel_offset

    @property
    def bbox(self) -> BoundingBox:
        """
        :getter: the bounding box in the big volume
        """
        bbox = BoundingBox.from_delta(self.voxel_offset, self.array.shape[-3:])
        return bbox

    @property
    def bounding_box(self) -> BoundingBox:
        return self.bbox

    @property
    def start(self) -> Cartesian:
        return self.bbox.start

    @property
    def stop(self) -> Cartesian:
        return self.bbox.stop

    @property
    def ndim(self) -> int:
        return self.array.ndim 

    @property 
    def shape(self) -> tuple:
        return self.array.shape 
    
    @property 
    def dtype(self) -> np.dtype:
        return self.array.dtype 

    @property
    def voxel_stop(self) -> tuple:
        return tuple(o + s for o, s in zip(self.voxel_offset, self.shape))

    def astype(self, dtype: Union[np.dtype, str]):
        if dtype is None:
            new_array = self.array
        elif dtype != self.array.dtype:
            new_array = self.array.astype(dtype)
        else:
            new_array = self.array
        return Chunk(new_array, voxel_offset=self.voxel_offset, voxel_size=self.voxel_size)

    def ascontiguousarray(self):
        new_array = np.ascontiguousarray(self.array)
        return Chunk(new_array, voxel_offset=self.voxel_offset, voxel_size=self.voxel_size)

    def max(self, *args, **kwargs):
        return self.array.max(*args, **kwargs)

    def min(self, *args, **kwargs):
        return self.array.min(*args, **kwargs)

    def shrink(self, size: tuple):
        """shrink the array from surrounding boundary

        Args:
            size (tuple): the shrink size in -Z,-Y, -X, Z, Y, X direction.
            sometimes, the shrinking might be asymmetric. that's why we need
            6 elements rather than 3.
        """
        assert len(size) == 6 or len(size) == 3
        z, y, x = self.shape[-3:]
        self.array = self.array[
            ...,
            size[0]:z-size[-3],
            size[1]:y-size[-2],
            size[2]:x-size[-1],
        ]
        self.voxel_offset += Cartesian.from_collection(size[:3])

    def transpose(self, only_array: bool=False):
        """To-Do: support arbitrary axis transpose"""
        new_array = self.array.transpose()
        if not only_array and self.voxel_offset is not None:
            voxel_offset = self.voxel_offset[::-1]
        else:
            voxel_offset = self.voxel_offset
        
        if not only_array and self.voxel_size is not None:
            voxel_size = self.voxel_size[::-1]
        else:
            voxel_size = self.voxel_size
        return Chunk(new_array, voxel_offset=voxel_offset, voxel_size=voxel_size)

    def fill(self, x):
        self.array.fill(x)

    def squeeze_channel(self, axis: int = 0) -> np.ndarray:
        """given a 4D array, squeeze the channel axis."""
        assert self.array.ndim == 4
        new_array = np.squeeze(self, axis=axis)
        return Chunk(new_array, voxel_offset=self.voxel_offset, voxel_size=self.voxel_size)

    # @profile(precision=0)
    def channel_voting(self):
        assert self.ndim == 4
        assert self.shape[0] <= 256
        out = np.empty(self.shape[1:], dtype=np.uint8)
        np.argmax(self.array, axis=0, out=out)
        # our selected channel index start from 1
        out += 1
        return Chunk(out, 
            voxel_offset=self.voxel_offset, 
            voxel_size=self.voxel_size,
            layer_type='segmentation',
        )

    def mask_using_last_channel(self, threshold: float = 0.3) -> np.ndarray:
        mask = (self.array[-1, :, :, :] < threshold)
        ret = self.array[:-1, ...]
        ret *= mask
        return Chunk(ret, voxel_offset=self.voxel_offset, voxel_size=self.voxel_size)

    def crop_margin(
            self, 
            margin_size: tuple = None, 
            output_bbox: BoundingBox=None
        ):
        """_summary_

        Args:
            margin_size (tuple, optional): -z,-y,-x,+z,+y,+x. Defaults to None.
            output_bbox (BoundingBox, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        if margin_size:
            sz,sy,sx = self.array.shape[-3:]
            if len(margin_size) == 3:
                new_array = self.array[...,
                    margin_size[0]: sz-margin_size[0],
                    margin_size[1]: sy-margin_size[1],
                    margin_size[2]: sx-margin_size[2]]
            elif len(margin_size) == 6:
                new_array = self.array[...,
                    margin_size[0]: sz-margin_size[3],
                    margin_size[1]: sy-margin_size[4],
                    margin_size[2]: sx-margin_size[5]]
            else:
                raise ValueError('only support 3 or 6 elements.')
            voxel_offset = tuple(
                o + m for o, m in zip(self.voxel_offset, margin_size))
            return Chunk(new_array, voxel_offset=voxel_offset, voxel_size=self.voxel_size)
        else:
            logging.info('automatically crop the chunk to output bounding box.')
            assert output_bbox is not None
            return self.cutout(output_bbox.slices)
    
    def threshold(self, threshold: float):
        array = self.array > threshold
        if array.ndim == 4:
            assert array.shape[0] == 1
            array = array[0, ...]
        seg = Chunk(array, voxel_offset = self.voxel_offset, voxel_size=self.voxel_size)
        # neuroglancer do not support bool datatype
        # numpy store bool as uint8 datatype, so this will not increase size.
        seg = seg.astype(np.uint8)
        return seg
    
    def where(self, mask: np.ndarray) -> tuple:
        """
        find the indexes of masked value as an alternative of np.where function

        :param mask: binary ndarray as mask
        :return: the coordinates in global coordinates.
        """
        isinstance(mask, np.ndarray)
        assert mask.shape == self.shape
        return tuple(i + o for i, o in zip(np.where(mask), self.voxel_offset))

    def add_overlap(self, other):
        """
        sum up overlaping region with another chunk

        :param other: another chunk
        :return: sum up result.
        """
        assert isinstance(other, Chunk)
        overlap_slices = self._get_overlap_slices(other.slices)
        self.array[overlap_slices] += other.array[overlap_slices]

    def cutout(self, x: Union[tuple, BoundingBox, Bbox]):
        """
        cutout a region of interest from this chunk

        :param slices: the global slices of region of interest
        :return: another chunk of region of interest
        """
        if isinstance(x, BoundingBox) or isinstance(x, Bbox):
            slices = x.slices
        else:
            slices = x
            
        if len(slices) == self.ndim - 1:
            slices = (slice(0, self.shape[0]), ) + slices
        internalSlices = self._get_internal_slices(slices)
        arr = self.array[internalSlices]
        voxel_offset = tuple(s.start for s in slices[-3:])
        return Chunk(arr, 
            voxel_offset=voxel_offset, 
            voxel_size=self.voxel_size, 
            layer_type=self.layer_type)

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
            zip(patch.slices, self.ndoffset, self.shape)
        )
        shape = (s.stop - s.start for s in internal_slices)
        patch_starts = (
            i.start - s.start + o for s, o, i in 
            zip(patch.slices, self.ndoffset, internal_slices)
        )
        patch_slices = tuple(slice(s, s+h) for s, h in zip(patch_starts, shape))

        self.array[internal_slices] += patch.array[patch_slices]

    def maskout(self, chunk: Chunk):
        """ Make part of the chunk to be black.
        """
        assert chunk.voxel_size is not None
        assert self.voxel_size is not None
        assert self.voxel_size >= chunk.voxel_size

        # the voxel size should be divisible
        assert Cartesian(0, 0, 0) == self.voxel_size % chunk.voxel_size

        factor = self.voxel_size // chunk.voxel_size
        for offset in np.ndindex(factor):
            chunk.array[
                ...,
                np.s_[offset[0]::factor[0]],
                np.s_[offset[1]::factor[1]],
                np.s_[offset[2]::factor[2]]] *= self.array

    def _get_overlap_slices(self, other_slices):
        return tuple(
            slice(max(s1.start, s2.start), min(s1.stop, s2.stop))
            for s1, s2 in zip(self.slices, other_slices))

    def _get_internal_slices(self, slices):
        assert self.ndim == len(slices)
        return tuple(
            slice(s.start - o, s.stop - o)
            for s, o in zip(slices, self.ndoffset))


    def validate(self):
        """validate the completeness of this chunk, there
        should not have black boxes.

        """
        validate_by_template_matching(self.array)

    def gaussian_filter_2d(self, sigma: Union[int, tuple, list] = 1):
        """gaussion filter for smoothing or blurring

        Args:
            sigma (Union[int, tuple, list]): the standard deviation of gaussian filter
        """
        for z in range(self.shape[-3]):
            self.array[..., z, :, :] = gaussian_filter(self.array[..., z, :, :], sigma)
