import unittest
import numpy as np
import os
import shutil

from chunkflow.chunk import Chunk
from chunkflow.flow.save_pngs import SavePNGsOperator


def read_write_h5(chunk):
    assert isinstance(chunk, np.ndarray)
    file_name = 'test.h5'
    if os.path.exists(file_name):
        os.remove(file_name)

    chunk.to_h5(file_name)

    chunk2 = Chunk.from_h5(file_name)
    assert np.alltrue(chunk == chunk2)
    assert chunk.global_offset == chunk2.global_offset
    os.remove(file_name)


def read_write_tif(chunk):
    """We'll lost global offset information using tif format!"""
    assert isinstance(chunk, np.ndarray)
    file_name = 'test.tif'
    if os.path.exists(file_name):
        os.remove(file_name)
    
    chunk.to_tif(file_name)
    chunk2 = Chunk.from_tif(file_name)
    assert np.alltrue(chunk == chunk2)
    os.remove(file_name)


def save_pngs(chunk):
    # test save images
    output_path = '/tmp/test/'
    save_pngs_operator = SavePNGsOperator(output_path)
    save_pngs_operator(chunk)
    print('remove the temporary directory.')
    shutil.rmtree(output_path)


class TestReadWrite(unittest.TestCase):
    def test_read_write_image(self):
        print('test image io...')
        arr = np.random.randint(0, 256, size=(8, 16, 16), dtype=np.uint8)
        chunk = Chunk(arr, global_offset=(1, 2, 3))
        read_write_h5(chunk)
        read_write_tif(chunk)

    def test_read_write_aff(self):
        print('test affinitymap io...')
        arr = np.random.rand(3, 8, 16, 16).astype(np.float32)
        chunk = Chunk(arr, global_offset=(0, 1, 2, 3))
        read_write_h5(chunk)


if __name__ == '__main__':
    unittest.main()
