import numpy as np

from cloudvolume.lib import Vec, Bbox

from .igneous.tasks import downsample_and_upload


def create_thumbnail(chunk, thumbnail_volume, chunk_mip):
    # transpose to xyzc
    chunk = np.transpose(chunk)

    # only use the last channel, it is the Z affinity if this is affinitymap
    chunk = chunk[:, :, :, -1]
    if chunk.dtype == np.float32: 
        image = (chunk * 255.0).astype(np.uint8)

    downsample_and_upload(
        image,
        Bbox.from_slices(chunk.slices),
        thumbnail_volume,
        Vec(*(chunk.shape[:3])),
        mip=chunk_mip,
        axis='z',
        skip_first=True,
        only_last_mip=True)
    # transpose back to czyx 
    chunk = np.transpose(chunk)    

