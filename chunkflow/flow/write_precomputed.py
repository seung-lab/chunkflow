import logging
import time
import os
import json
import numpy as np

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox, yellow
from cloudvolume.storage import Storage

from chunkflow.lib.igneous.tasks import downsample_and_upload
from chunkflow.chunk import Chunk

from .base import OperatorBase
from chunkflow.lib.igneous.tasks import downsample_and_upload
#from .downsample_upload import DownsampleUploadOperator


class WritePrecomputedOperator(OperatorBase):
    def __init__(self,
                 volume_path: str,
                 mip: int,
                 upload_log: bool = True,
                 create_thumbnail: bool = False,
                 name: str = 'save'):
        super().__init__(name=name)
        
        self.upload_log = upload_log
        self.create_thumbnail = create_thumbnail
        self.mip = mip

        # if not volume_path.startswith('precomputed://'):
        #     volume_path += 'precomputed://'
        self.volume_path = volume_path
        
        # gevent.monkey.patch_all(thread=False)
        self.volume = CloudVolume(
            self.volume_path,
            fill_missing=True,
            bounded=False,
            autocrop=True,
            mip=self.mip,
            cache=False,
            green_threads=True,
            progress=True)

        if upload_log:
            log_path = os.path.join(volume_path, 'log')
            self.log_storage = Storage(log_path)

    def create_chunk_with_zeros(self, bbox, num_channels, dtype):
        """Create a fake all zero chunk. 
        this is used in skip some operation based on mask."""
        shape = (num_channels, *bbox.size3())
        arr = np.zeros(shape, dtype=dtype)
        chunk = Chunk(arr, voxel_offset=(0, *bbox.minpt))
        return chunk

    def __call__(self, chunk, log=None):
        assert isinstance(chunk, Chunk)
        logging.info('save chunk.')
        
        start = time.time()
        

        chunk = self._auto_convert_dtype(chunk, self.volume)
        
        # transpose czyx to xyzc order
        arr = np.transpose(chunk.array)
        self.volume[chunk.slices[::-1]] = arr
        
        if self.create_thumbnail:
            self._create_thumbnail(chunk)

        # add timer for save operation itself
        if log:
            log['timer'][self.name] = time.time() - start

        if self.upload_log:
            self._upload_log(log, chunk.bbox)

    def _auto_convert_dtype(self, chunk, volume):
        """convert the data type to fit volume datatype"""
        if volume.dtype != chunk.dtype:
            print(yellow(f'converting chunk data type {chunk.dtype} ' + 
                         f'to volume data type: {volume.dtype}'))
            # float_chunk = chunk.astype(np.float64)
            # chunk = float_chunk / np.iinfo(chunk.dtype).max * np.iinfo(self.volume.dtype).max
            chunk = chunk / chunk.array.max() * np.iinfo(volume.dtype).max
            return chunk.astype(volume.dtype)
        else:
            return chunk

    def _create_thumbnail(self, chunk):
        logging.info('creating thumbnail...')

        thumbnail_layer_path = os.path.join(self.volume_path, 'thumbnail')
        thumbnail_volume = CloudVolume(
            thumbnail_layer_path,
            compress='gzip',
            fill_missing=True,
            bounded=False,
            autocrop=True,
            mip=self.mip,
            cache=False,
            green_threads=True,
            progress=False)

        # only use the last channel, it is the Z affinity
        # if this is affinitymap
        image = chunk[-1, :, :, :]
        if np.issubdtype(image.dtype, np.floating):
            image = (image * 255).astype(np.uint8)
        
        #self.thumbnail_operator(image)
        # transpose to xyzc
        image = np.transpose(image)
        image_bbox = Bbox.from_slices(chunk.slices[::-1][:3])

        downsample_and_upload(image,
                              image_bbox,
                              thumbnail_volume,
                              Vec(*(image.shape)),
                              mip=self.mip,
                              max_mip=6,
                              axis='z',
                              skip_first=True,
                              only_last_mip=True)

    def _upload_log(self, log, output_bbox):
        assert log
        assert isinstance(output_bbox, Bbox)

        logging.info(f'uploaded log: {log}')

        # write to google cloud storage
        self.log_storage.put_file(file_path=output_bbox.to_filename() +
                                  '.json',
                                  content=json.dumps(log),
                                  content_type='application/json')
