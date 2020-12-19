import logging
from .base import OperatorBase
from cloudvolume import CloudVolume
import tinybrain
import numpy as np
from cloudvolume.lib import Bbox

class DownsampleUploadOperator(OperatorBase):
    """
    Multiple mip level downsampling including image and segmenation.

    For image, the algorithm will be automatically choosen as average pooling.
    For segmentation, the algorithm will be Will Silversman's countless algorithm to perform model pooling. The most frequent segmentation ID will be choosen.

    The type of chunk was automatically determined from the data type. 
    Image: uint8, floating
    Segmentation: uint16, uint32, uint64,...
    """
    def __init__(self,
                 volume_path: str,
                 factor: tuple = (2,2,2),
                 chunk_mip: int = 0,
                 start_mip: int = None,
                 stop_mip: int = 5,
                 fill_missing: bool = True,
                 name='downsample-upload'):
        """
        volume_path: (str) path of volume
        factor: (tuple) downsampling factor in z,y,x order
        chunk_mip: (int) the mip level of input chunk
        start_mip: (int) the mip level for starting uploading
        stop_mip: (int) the mip level for stoping uploading. Note that the indexing follows python indexing, this stop mip will not be included. For example, if you would like to upload mip level 1 to 4, the start mip will be 1, and the stop mip should be 5.
        fill_missing: (bool) fill missing blocks with zeros or not. See same parameter in cloudvolume.
        """
        super().__init__(name=name)
        
        if start_mip is None:
            start_mip = chunk_mip + 1

        verbose = (logging.getLogger().getEffectiveLevel() <= 30)
        vols = dict()
        for mip in range(start_mip, stop_mip):
            vols[mip] = CloudVolume(volume_path,
                                    fill_missing=fill_missing,
                                    bounded=False,
                                    autocrop=True,
                                    mip=mip,
                                    green_threads=True,
                                    progress=verbose)

        self.vols = vols
        self.factor = factor
        self.chunk_mip = chunk_mip
        self.start_mip = start_mip
        self.stop_mip = stop_mip
        assert len(factor) == 3
        assert np.alltrue([f>=1 for f in factor])
        assert stop_mip > start_mip
        assert start_mip > chunk_mip

    def __call__(self, chunk):
        assert 3 == chunk.ndim 
        voxel_offset = chunk.voxel_offset

        num_mips = self.stop_mip - self.chunk_mip
        # tinybrain use F order and require 4D array!
        chunk2 = np.transpose(chunk)
        # chunk2 = np.reshape(chunk2, (*chunk2.shape, 1))
        chunk2 = np.expand_dims(chunk2, 3)

        if np.issubdtype(chunk.dtype, np.floating) or chunk.dtype == np.uint8:
            pyramid = tinybrain.downsample_with_averaging(chunk2,
                                                          factor=self.factor[::-1],
                                                          num_mips=num_mips)
        else:
            pyramid = tinybrain.downsample_segmentation(chunk2,
                                                        factor=self.factor[::-1],
                                                        num_mips=num_mips)

        for mip in range(self.start_mip, self.stop_mip):
            # the first chunk in pyramid is already downsampled!
            downsampled_chunk = pyramid[mip - self.chunk_mip - 1]
            # compute new offset, only downsample the y,x dimensions
            offset = np.divide(voxel_offset, np.asarray([
                self.factor[0]**(mip - self.chunk_mip), 
                self.factor[1]**(mip - self.chunk_mip), 
                self.factor[2]**(mip - self.chunk_mip)]))
            bbox = Bbox.from_delta(offset, downsampled_chunk.shape[0:3][::-1])
            # upload downsampled chunk, note that we should use F order in the indexing
            self.vols[mip][bbox.to_slices()[::-1]] = downsampled_chunk
