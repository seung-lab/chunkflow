import numpy as np
from chunkflow.chunk import Chunk
from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
import tensorstore as ts


def execute(bbox: BoundingBox, driver: str=None, kvstore: str=None, cache: int=None, 
        voxel_size: tuple=None, output_name: str='chunk'):
    """Load chunk from dataset using tensorstore"""
    if driver == 'n5':
        kv_driver, path = kvstore.split('://')
        kvstore = {
            'driver': kv_driver,
            'path': path
        }
    dataset_future = ts.open({
        'driver': driver,
        'kvstore': kvstore,
        'context': {
            'cache_pool': {
                'total_bytes_limit': cache,
            }
        },
        'recheck_cached_data': 'open',
    })
    dataset = dataset_future.result()

    slices = bbox.slices
    arr = dataset[slices[0], slices[1], slices[2]].read().result()
    assert arr.ndim == 4
    arr = arr.transpose()
    if arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    chunk = Chunk(arr, 
        voxel_offset = bbox.start, 
        voxel_size = Cartesian.from_collection(voxel_size),
    )
    return chunk
