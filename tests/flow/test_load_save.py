import unittest
import numpy as np
import os
import shutil

from chunkflow.chunk import Chunk
from chunkflow.lib.cartesian_coordinate import Cartesian

from chunkflow.flow.save_pngs import SavePNGsOperator


def load_save_h5(chunk):
    assert isinstance(chunk, Chunk)
    file_name = 'test.h5'
    if os.path.exists(file_name):
        os.remove(file_name)

    chunk.to_h5(file_name, chunk_size=None)

    chunk2 = Chunk.from_h5(file_name)
    assert chunk == chunk2
    assert chunk.voxel_offset == chunk2.voxel_offset
    os.remove(file_name)


def load_save_tif(chunk):
    """We'll lost global offset information using tif format!"""
    assert isinstance(chunk, Chunk)
    file_name = 'test.tif'
    if os.path.exists(file_name):
        os.remove(file_name)
    
    chunk.to_tif(file_name)
    chunk2 = Chunk.from_tif(file_name)
    
    # we can not preserve the global offset here
    # so chunk2's global offset will all be 0
    np.testing.assert_array_equal(chunk.array, chunk2.array)
    os.remove(file_name)


def save_pngs(chunk):
    # test save images
    output_path = '/tmp/test/'
    save_pngs_operator = SavePNGsOperator(output_path)
    save_pngs_operator(chunk)
    print('remove the temporary directory.')
    shutil.rmtree(output_path)


class TestReadSave(unittest.TestCase):
    def test_load_save_image(self):
        print('test image io...')
        arr = np.random.randint(0, 256, size=(8, 16, 16), dtype=np.uint8)
        chunk = Chunk(arr, voxel_offset=Cartesian(1, 2, 3))
        load_save_h5(chunk)
        load_save_tif(chunk)

    def test_load_save_aff(self):
        print('test affinitymap io...')
        arr = np.random.rand(3, 8, 16, 16).astype(np.float32)
        chunk = Chunk(arr, voxel_offset=Cartesian(1, 2, 3))
        load_save_h5(chunk)


if __name__ == '__main__':
    unittest.main()
