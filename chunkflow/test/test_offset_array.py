import numpy as np
import unittest
from chunkflow.offset_array import OffsetArray


class TestOffsetArray(unittest.TestCase):
    def test_normalize(self):
        print('\ntest offset array normalization...')
        arr = np.ones((3, 3, 3), dtype='float32')
        chunk = OffsetArray(arr, (-1, -1, -1))
        mask = np.ones((3, 3, 3), dtype='float32') * 0.5
        chunk *= mask
        self.assertTrue(
            np.all(chunk == np.ones((3, 3, 3), dtype='float32') * 0.5))

    def test_slices(self):
        arr = np.ones((3, 3, 3), dtype='float32')
        chunk = OffsetArray(arr, (-1, -1, -1))
        self.assertEqual(chunk.slices,
                         (slice(-1, 2), slice(-1, 2), slice(-1, 2)))

    def test_where(self):
        arr = np.asarray([0.1, 0.7])
        selected1 = np.where(arr > 0.5)
        chunk = OffsetArray(arr, (-1, ))
        selected2 = chunk.where(chunk > 0.5)
        for i1, i2 in zip(selected1, selected2):
            # print('i1: {}, i2: {}'.format(i1, i2))
            self.assertTrue((i1 - i2 == 1).all())


if __name__ == '__main__':
    unittest.main()
