from chunkflow.chunk import Chunk
from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
import zarr


def execute(bbox: BoundingBox, driver: str = 'n5', 
        voxel_size: Cartesian = None,
        n5_dir: str = None, 
        group_path: str = None):
    """load zarr volume"""
    if driver == 'n5':
        fsstore = zarr.N5FSStore(n5_dir, anon=True)
    else:
        raise ValueError('unsupported driver type.')

    group = zarr.open(fsstore)
    breakpoint()
    zdata = group[group_path]
    arr = zdata[bbox.to_slices()]

    chunk = Chunk(arr, voxel_offset=bbox.start, voxel_size=voxel_size)

    return chunk