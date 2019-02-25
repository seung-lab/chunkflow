import numpy as np

from cloudvolume.lib import Vec, Bbox
from cloudvolume import CloudVolume

from .igneous.tasks import downsample_and_upload


def create_thumbnail(chunk, thumbnail_volume_path, chunk_mip,
                     verbose=True):
    if verbose:
        print('create thumbnail in {}'.format(thumbnail_volume_path) )

    thumbnail_volume = CloudVolume(
        thumbnail_volume_path,
        compress='gzip',
        fill_missing=True,
        bounded=False,
        autocrop=True,
        mip=chunk_mip,
        progress=verbose)

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
        mip=chunk_mip,
        axis='z',
        skip_first=True,
        only_last_mip=True)

    # chunk should not be changed
    return chunk
