import numpy as np
import unittest
from cloudvolume.lib import Bbox
from chunkflow.chunk import Chunk


class TestChunk(unittest.TestCase):
    def setUp(self):
        self.size = (3, 3, 3)
        self.global_offset = (-1, -1, -1)
        arr = np.ones(self.size, dtype='float32')
        self.chunk = Chunk(arr, self.global_offset)

    def test_bbox(self):
        self.assertEqual(self.chunk.bbox,
                         Bbox.from_delta(self.global_offset, self.size))

    def test_slices(self):
        self.assertEqual(self.chunk.slices,
                         (slice(-1, 1), slice(-1, 1), slice(-1, 1)))

    def test_initialization(self):
        arr = np.ones((3, 3, 3), dtype='float32')
        chunk = Chunk(arr)
        (self.assertEqual(o, 0) for o in chunk.global_offset)

    def test_attr(self):
        """
        attribute should not change after numpy operation
        """
        chunk2 = np.sin(self.chunk)
        self.assertIsInstance(chunk2, Chunk)
        self.assertEqual(chunk2.global_offset, self.chunk.global_offset)

        chunk2 = self.chunk * 255
        self.assertIsInstance(chunk2, Chunk)
        self.assertEqual(chunk2.global_offset, self.chunk.global_offset)

        chunk2 = chunk2.astype(np.uint8)
        self.assertIsInstance(chunk2, Chunk)
        self.assertEqual(chunk2.global_offset, self.chunk.global_offset)

        chunk2 = np.transpose(chunk2)
        print('type of chunk after transpose: {}'.format(type(chunk2)))
        self.assertIsInstance(chunk2, Chunk)
        self.assertEqual(chunk2.global_offset, self.chunk.global_offset)

        #chunk2 = np.ascontiguousarray(chunk2)
        #self.assertIsInstance(chunk2, Chunk)
        #self.assertEqual(chunk2.global_offset, self.chunk.global_offset)

        #chunk2 = np.pad(chunk2, ((0,2),(0,2),(0,2)), 'reflect')
        #self.assertIsInstance(chunk2, Chunk)
        #self.assertEqual(chunk2.global_offset, self.chunk.global_offset)

    def test_normalize(self):
        print('\ntest chunk normalization...')
        arr = np.ones((3, 3, 3), dtype='float32')
        chunk = Chunk(arr, (-1, -1, -1))

        mask = np.ones((3, 3, 3), dtype='float32') * 0.5
        chunk *= mask
        self.assertTrue(np.alltrue(chunk == mask))

    def test_slices(self):
        arr = np.ones((1, 3, 3, 3), dtype='float32')
        chunk = Chunk(arr, (0, -1, -1, -1))
        self.assertEqual(
            chunk.slices,
            (slice(0, 1), slice(-1, 2), slice(-1, 2), slice(-1, 2)))

    def test_where(self):
        arr = np.asarray([0.1, 0.7])
        selected1 = np.where(arr > 0.5)
        chunk = Chunk(arr, (-1, ))
        selected2 = chunk.where(chunk > 0.5)
        for i1, i2 in zip(selected1, selected2):
            # print('i1: {}, i2: {}'.format(i1, i2))
            self.assertTrue((i1 - i2 == 1).all())


if __name__ == '__main__':
    unittest.main()
