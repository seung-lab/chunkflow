import numpy as np

from cloudvolume.lib import Vec, Bbox
from cloudvolume import CloudVolume

from chunkflow.igneous.tasks import downsample_and_upload
from .operator_base import OperatorBase


class CreateThumbnailOperator(OperatorBase):
    def __init__(self, volume_path,
                 chunk_mip=0, verbose=True,
                 name='create-thumbnail'):
        super().__init__(name=name)
        self.volume_path = volume_path
        self.chunk_mip = chunk_mip
        self.verbose = verbose

    def __call__(self, chunk):
        if self.verbose:
            print('create thumbnail in {}'.format(self.volume_path))

        thumbnail_volume = CloudVolume(
            self.volume_path,
            compress='gzip',
            fill_missing=True,
            bounded=False,
            autocrop=True,
            mip=self.chunk_mip,
            progress=self.verbose)

        # only use the last channel, it is the Z affinity
        # if this is affinitymap
        image = chunk[-1, :, :, :]
        if np.issubdtype(image.dtype, np.floating):
            image = (image * 255).astype(np.uint8)

        # transpose to xyzc
        image = np.transpose(image)
        image_bbox = Bbox.from_slices(chunk.slices[::-1][:3])
        
        downsample_and_upload(
            image,
            image_bbox,
            thumbnail_volume,
            Vec(*(image.shape)),
            mip=self.chunk_mip,
            axis='z',
            skip_first=True,
            only_last_mip=True)

        # chunk should not be changed
        return chunk
