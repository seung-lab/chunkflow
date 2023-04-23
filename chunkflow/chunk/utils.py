import numpy as np

from chunkflow.chunk import Chunk
from chunkflow.volume import PrecomputedVolume
from chunkflow.lib.cartesian_coordinate import Cartesian
from chunkflow.flow.plugin import str_to_dict 


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
    