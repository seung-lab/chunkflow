import unittest
import shutil
import numpy as np
from chunkflow.chunk import Chunk

from cloudvolume import CloudVolume
from cloudvolume.storage import Storage
from cloudvolume.lib import generate_random_string, Bbox

from chunkflow.flow.downsample_upload import DownsampleUploadOperator


class TestDownsampleUpload(unittest.TestCase):
    def test_downsample_upload(self):
        print('test downsample and upload...')
        # compute parameters
        mip = 0
        size = (16, 512, 512)

        # create image dataset using cloud-volume
        img = np.random.randint(np.iinfo(np.uint32).max, size=size)
        img = img.astype(np.uint32)
        chunk = Chunk(img, global_offset=[2, 32, 32])
        # save the input to disk
        volume_path = 'file:///tmp/test/cutout/' + generate_random_string()
        CloudVolume.from_numpy(
            np.transpose(img),
            vol_path=volume_path,
            voxel_offset=(32, 32, 2),
            chunk_size=(32, 32, 4),
            max_mip=4,
            layer_type='segmentation')

        operator = DownsampleUploadOperator(
            volume_path, input_mip=0, start_mip=1, stop_mip=4)
        operator(chunk)
        shutil.rmtree('/tmp/test')


if __name__ == '__main__':
    unittest.main()
