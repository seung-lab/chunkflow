import shutil
import tempfile
import numpy as np
from chunkflow.chunk import Chunk

from cloudvolume import CloudVolume
from cloudvolume.storage import Storage
from cloudvolume.lib import Bbox

from chunkflow.flow.downsample_upload import DownsampleUploadOperator


def hierarchical_downsample(chunk, layer_type='segmentation'):
    # save the input to disk
    tempdir = tempfile.mkdtemp()
    volume_path = 'file://' + tempdir
    CloudVolume.from_numpy(chunk.transpose(),
                           vol_path=volume_path,
                           voxel_offset=(32, 32, 2),
                           chunk_size=(32, 32, 4),
                           max_mip=4,
                           layer_type=layer_type)

    operator = DownsampleUploadOperator(volume_path,
                                        chunk_mip=0,
                                        start_mip=1,
                                        stop_mip=4)
    operator(chunk)
    shutil.rmtree(tempdir)

def test_segmentation():
    print('test downsample and upload...')
    # compute parameters
    mip = 0
    size = (16, 512, 512)

    # create image dataset using cloud-volume
    img = np.random.randint(np.iinfo(np.uint32).max, 
                            size=size, dtype=np.uint32)
    chunk = Chunk(img, global_offset=[2, 32, 32])
    hierarchical_downsample(chunk)
    
def test_image():
    print('test downsample and upload...')
    # compute parameters
    mip = 0
    size = (16, 512, 512)

    # create image dataset using cloud-volume
    img = np.random.randint(np.iinfo(np.uint8).max, 
                            size=size, dtype=np.uint8)
    chunk = Chunk(img, global_offset=[2, 32, 32])
    hierarchical_downsample(chunk, layer_type='image')
    
def test_psd_map():
    print('test downsample and upload...')
    # compute parameters
    mip = 0
    size = (16, 512, 512)

    # create image dataset using cloud-volume
    img = np.random.rand(*size).astype(np.float32)
    chunk = Chunk(img, global_offset=[2, 32, 32])
    hierarchical_downsample(chunk, layer_type='image')
