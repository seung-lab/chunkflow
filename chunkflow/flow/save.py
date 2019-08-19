import time
import os
import json
import numpy as np

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox
from cloudvolume.storage import Storage

from chunkflow.lib.igneous.tasks import downsample_and_upload
from chunkflow.chunk import Chunk

from .base import OperatorBase


class SaveOperator(OperatorBase):
    def __init__(self,
                 volume_path: str,
                 mip: int,
                 upload_log: bool = True,
                 create_thumbnail: bool = False,
                 nproc: int = 0,
                 verbose: bool = True,
                 name: str = 'save'):
        super().__init__(name=name, verbose=verbose)
        if nproc < 0:
            nproc = True
        elif nproc == 0:
            nproc = False

        self.upload_log = upload_log
        self.create_thumbnail = create_thumbnail
        self.mip = mip

        self.volume = CloudVolume(volume_path,
                                  fill_missing=True,
                                  bounded=False,
                                  autocrop=True,
                                  mip=mip,
                                  parallel=nproc,
                                  progress=verbose)

        if upload_log:
            log_path = os.path.join(volume_path, 'log')
            self.log_storage = Storage(log_path)

        if create_thumbnail:
            self.thumbnail_volume = CloudVolume(os.path.join(
                volume_path, 'thumbnail'),
                                                compress='gzip',
                                                fill_missing=True,
                                                bounded=False,
                                                autocrop=True,
                                                mip=mip,
                                                progress=verbose)

    def create_chunk_with_zeros(self, bbox):
        """Create a fake all zero chunk"""
        shape = (self.volume.num_channels, *bbox.size3())
        arr = np.zeros(shape, dtype=self.volume.dtype)
        chunk = Chunk(arr, global_offset=(0, *bbox.minpt))
        return chunk

    def __call__(self, chunk, log={'timer': {}}, output_bbox=None):
        start = time.time()

        chunk = self._auto_convert_dtype(chunk)

        chunk_slices = chunk.slices
        # transpose czyx to xyzc order
        arr = np.transpose(chunk)
        self.volume[chunk_slices[::-1]] = arr

        if self.create_thumbnail:
            self._create_thumbnail(chunk)

        # add timer for save operation itself
        log['timer'][self.name] = time.time() - start
        if self.upload_log:
            if output_bbox is None:
                output_bbox = Bbox.from_delta((0, 0, 0), chunk.shape)
            self._upload_log(log, output_bbox)

    def _auto_convert_dtype(self, chunk):
        """convert the data type to fit volume datatype"""
        if self.volume.dtype != chunk.dtype:
            float_chunk = chunk.astype(np.float64)
            #chunk = float_chunk / np.iinfo(chunk.dtype).max * np.iinfo(self.volume.dtype).max
            chunk = float_chunk / np.max(chunk) * np.iinfo(
                self.volume.dtype).max
            return chunk.astype(self.volume.dtype)
        else:
            return chunk

    def _create_thumbnail(self, chunk):
        if self.verbose:
            print('creating thumbnail...')
        # only use the last channel, it is the Z affinity
        # if this is affinitymap
        image = chunk[-1, :, :, :]
        if np.issubdtype(image.dtype, np.floating):
            image = (image * 255).astype(np.uint8)

        # transpose to xyzc
        image = np.transpose(image)
        image_bbox = Bbox.from_slices(chunk.slices[::-1][:3])

        downsample_and_upload(image,
                              image_bbox,
                              self.thumbnail_volume,
                              Vec(*(image.shape)),
                              mip=self.mip,
                              axis='z',
                              skip_first=True,
                              only_last_mip=True)

    def _upload_log(self, log, output_bbox):
        assert log
        assert isinstance(output_bbox, Bbox)

        if self.verbose:
            print('uploaded log: ', log)

        # write to google cloud storage
        self.log_storage.put_file(file_path=output_bbox.to_filename() +
                                  '.json',
                                  content=json.dumps(log),
                                  content_type='application/json')
