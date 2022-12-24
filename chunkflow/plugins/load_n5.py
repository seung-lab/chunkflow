
import zarr

import boto3
from botocore import UNSIGNED
from botocore.client import Config
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

from chunkflow.chunk import Chunk
from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian

def execute(bbox: BoundingBox, 
        n5_dir: str = None, 
        group_path: str = None,
        voxel_size: tuple = None,
        type: str = None,
        driver: str = 'n5', ):
    # container_url = os.path.join(n5_dir)
    # only n5 driver is supported for now
    assert driver == 'n5'
    if isinstance(voxel_size, tuple):
        voxel_size = Cartesian.from_collection(voxel_size)

    fsstore = zarr.N5FSStore(n5_dir, anon=True)
    img_zarr = zarr.open(fsstore, path=group_path)
    img_arr = img_zarr[bbox.slices]

    img_chk = Chunk(img_arr, 
        voxel_offset = bbox.start, 
        voxel_size = voxel_size,
        type = type
        )
    
    return img_chk
    

