from __future__ import annotations
from typing import Union
from abc import ABC, abstractmethod, abstractproperty
from functools import cached_property
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from cloudvolume import CloudVolume
from chunkflow.lib.utils import str_to_dict
from .lib.cartesian_coordinate import \
    BoundingBox, Cartesian, BoundingBoxes, PhysicalBoudingBox
from .chunk import Chunk


@dataclass(frozen=True)
class AbstractVolume(ABC):

    @abstractproperty
    def bounding_box(self) -> BoundingBox:
        pass

    @abstractproperty
    def block_size(self) -> Cartesian:
        pass

    @abstractproperty
    def voxel_size(self) -> Cartesian:
        pass

    @abstractmethod
    def cutout(self, key: Union[BoundingBox, list]) -> Chunk:
        pass

    @abstractmethod
    def save(self, chk: Chunk) -> None:
        pass

@dataclass(frozen=True)
class PrecomputedVolume(AbstractVolume):
    """The major difference with CloudVolume is that we use C order here. 
    ZYX indexing.

    Args:
        CloudVolume (class): the cloud-volume class
    """
    vol: CloudVolume
    
    @classmethod
    def from_cloudvolume_path(cls, path: str, *arg, 
            fill_missing: bool=True, **kwargs) -> PrecomputedVolume:
        """load from a cloud volume path
        This path could be encoded with keywords.
        For example: precomputed://file:///volume/path#preload=True;cache='/tmp/'

        Args:
            path (str): the volume path with keyword encoding.

        Returns:
            PrecomputedVolume: the Volume instance
        """
        if '#' in path:
            secs = path.split('#')
            assert len(secs) == 2
            path = secs[0]
            specified_keywords = str_to_dict(secs[1])
            if 'preload' in specified_keywords:
                del specified_keywords['preload']
            kwargs.update(specified_keywords)

        vol = CloudVolume(path, *arg, fill_missing=fill_missing, **kwargs)
        return cls(vol)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, vol_path: str) -> PrecomputedVolume:
        vol = CloudVolume.from_numpy(np.transpose(arr), vol_path=vol_path)
        return cls(vol)

    @cached_property
    def bounding_box(self):
        bbox = self.vol.bounds
        bbox = BoundingBox.from_bbox(bbox)
        # from xyz to zyx
        bbox = bbox.inverse_order()
        return bbox
    
    @cached_property
    def start(self) -> Cartesian:
        return self.bounding_box.start

    @cached_property
    def stop(self) -> Cartesian:
        return self.bounding_box.stop

    @cached_property
    def voxel_size(self) -> Cartesian:
        voxel_size = self.vol.resolution[::-1]
        voxel_size = Cartesian.from_collection(voxel_size)
        return voxel_size

    @cached_property
    def dtype(self):
        return self.vol.dtype

    @cached_property
    def block_size(self):
        return Cartesian.from_collection(
            self.vol.chunk_size[::-1])

    @cached_property 
    def physical_bounding_box(self) -> PhysicalBoudingBox:
        return PhysicalBoudingBox(
            self.start, self.stop, self.voxel_size)

    @cached_property
    def block_bounding_boxes(self) -> BoundingBoxes:
        return self.bounding_box.decompose(self.block_size)

    # @property
    # def nonzero_block_bounding_boxes(self) -> BoundingBoxes:
    #     nnz_bboxes = []
    #     for bbox in tqdm(
    #             self.block_bounding_boxes, 
    #             desc='iterate mask blocks...'):
    #         block = self.cutout(bbox)
    #         if np.all(block > 0):
    #             nnz_bboxes.append(bbox)

    #     return nnz_bboxes

    
   
    @cached_property
    def shape(self):
        return self.vol.shape[::-1]

    def cutout(self, key: Union[BoundingBox, list]):
        if isinstance(key, BoundingBox):
            arr = self.vol[ key.slices[::-1] ]
            voxel_offset = key.start
        elif isinstance(key, list):
            arr = self.vol[key[::-1]]
            voxel_offset = Cartesian(key[0].start, key[1].start, key[2].start)
        else:
            raise ValueError('we only support BoundingBox or a list of slices')

        # transpose
        arr = np.transpose(arr)
        arr = np.asarray(arr)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        chunk = Chunk(arr, voxel_offset=voxel_offset, voxel_size=self.voxel_size) 
        return chunk

    def _auto_convert_dtype(self, chunk: Chunk):
        """convert the data type to fit volume datatype"""
        if np.issubdtype(self.dtype, np.floating) and \
                np.issubdtype(chunk.dtype, np.uint8):
            chunk = chunk.astype(self.dtype)
            chunk /= 255.
            # chunk = chunk / chunk.array.max() * np.iinfo(volume.dtype).max
        elif np.issubdtype(self.dtype, np.uint8) and \
                np.issubdtype(chunk.dtype, np.floating):
            chunk.max() <= 1.
            chunk *= 255

        if self.dtype != chunk.dtype:
            print(f'converting chunk data type {chunk.dtype} ' + 
                         f'to volume data type: {self.dtype}')
            return chunk.astype(self.dtype)
        else:
            return chunk

    def save(self, chunk: Chunk):
        chunk = self._auto_convert_dtype(chunk)
        
        # transpose czyx to xyzc order
        arr = np.transpose(chunk.array)
        self.vol[chunk.slices[::-1]] = arr

    def has_all_blocks(self, bbox: BoundingBox) -> bool:
        """the volume has all the blocks inside a bounding box or not

        Args:
            bbox (BoundingBox): region of interest

        Returns:
            result (bool): result
        """
        block_size = Cartesian.from_collection(self.vol.chunk_size)
        assert bbox.is_aligned_with(block_size)
        # crop the out size range
        bbox = bbox.intersection( self.bounding_box )
        block_list = self.vol.exists(bbox.cloud_volume_bbox_xyz)
        block_nums = (bbox // block_size).shape
        return np.prod(block_nums) == len(block_list)



# class ZarrVolume(AbstractVolume):

# class SynapseVolume:

def load_chunk_or_volume(file_path: str, *arg, **kwargs):
    """load chunk or volume
    Note that if the mip level is larger than 1, the volume will be preloaded to RAM as a chunk!

    Args:
        file_path (str): the file path of chunk or volume. 
    
    Returns:
        Union[Chunk, AbstractVolume]: loaded chunk or volume
    """
    if kwargs is None:
        kwargs = dict()

    if file_path.endswith('.h5'):
        return Chunk.from_h5(file_path)
    elif file_path.endswith('.npy'):
        arr = np.loads(file_path)
        return Chunk(array=arr)
    elif 'file://' in file_path:
        # Neuroglancer Precomputed images
        if '#' in file_path:
            file_path, kwarg_str = file_path.split('#')
            kwargs.update(str_to_dict(kwarg_str))
            assert len(kwargs) > 0
        if 'preload' in kwargs:
            preload = kwargs['preload']
            del kwargs['preload']
        else:
            preload = False
        
        vol = PrecomputedVolume.from_cloudvolume_path(file_path, *arg, **kwargs)
        if not preload:
            return vol
        else:
            chunk = vol.cutout(vol.bounding_box)
            chunk.voxel_size = vol.voxel_size
            return chunk
    else:
        raise ValueError(f'only .h5 and .npy files are supported, but got {file_path}')
    

def get_candidate_block_bounding_boxes_with_different_voxel_size(
        chunk: Union[Chunk, AbstractVolume], 
        voxel_size_low: Cartesian, 
        block_size_low: Cartesian = None) -> BoundingBoxes:
    """get candidate block bounding boxes with different voxel size.
    This is normally used to get a list of block bounding boxes for image that contains nonzero in a mask volume with higher mip level.

    Args:
        voxel_size_low (Cartesian): voxel size in lower mip level. The resolution is higher and voxel size is smaller.
        block_size_low (Cartesian): block size in low mip level.

    Returns:
        BoundingBoxes: bounding boxes that in a lower mip level with mask all nonzero.
    """
    # we only support lower mip levels for now
    assert voxel_size_low <= chunk.voxel_size, \
        f'expecting new voxel size smaller than current one: {voxel_size_low}<={chunk.voxel_size}'
    # assume that the block size is the same
    if block_size_low is None:
        assert isinstance(chunk, AbstractVolume)
        block_size_low = chunk.block_size
    nonzero_bboxes_low = BoundingBoxes()
    for bbox_high in tqdm(
            chunk.block_bounding_boxes, 
            desc='iterate mask blocks...'):
        block_high = chunk.cutout(bbox_high)
        if np.all(block_high == 0):
            continue
        pbbox_low = block_high.physical_bounding_box.to_other_voxel_size(
            voxel_size_low
        )
        for bbox_low in pbbox_low.decompose(block_size_low):
            pbbox_low = PhysicalBoudingBox(
                bbox_low.start, bbox_low.stop, voxel_size_low)
            pbbox_high = pbbox_low.to_other_voxel_size(chunk.voxel_size)
            chunk_high = block_high.cutout(pbbox_high)
            
            if np.any(chunk_high > 0):
                nonzero_bboxes_low.append(pbbox_low)

    return nonzero_bboxes_low