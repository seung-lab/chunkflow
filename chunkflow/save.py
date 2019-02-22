import numpy as np
from cloudvolume import CloudVolume 


def save(chunk, volume_path, mip, show_progress=False):
    output_volume = CloudVolume(
        volume_path,
        fill_missing=True,
        bounded=False,
        autocrop=True,
        mip=mip,
        progress=show_progress)
    
    chunk_slices = chunk.slices 
    # transpose czyx to xyzc order
    arr = np.transpose(chunk)
    output_volume[chunk_slices[::-1]] = arr 
