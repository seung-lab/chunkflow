import numpy as np

from chunkflow.chunk import Chunk
from chunkflow.volume import PrecomputedVolume
from chunkflow.flow.plugin import str_to_dict 


def load_chunk_or_volume(file_path: str, *arg, **kwargs):
    if file_path.endswith('.h5'):
        return Chunk.from_h5(file_path)
    elif file_path.endswith('.npy'):
        arr = np.loads(file_path)
        return Chunk(array=arr)
    elif file_path.startswith('file://'):
        # Neuroglancer Precomputed images
        if ':' in file_path:
            file_path, kwarg_str = file_path.split(':')
            if kwargs is None:
                kwargs = dict()
            kwargs = kwargs.update(str_to_dict(kwarg_str))
        return PrecomputedVolume.from_cloudvolume_path(file_path, *arg, **kwargs)
    else:
        raise ValueError(f'only .h5 and .npy files are supported, but got {file_path}')
    