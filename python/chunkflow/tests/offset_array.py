import numpy as np
import unittest
from chunkflow.worker.offset_array import OffsetArray


class TestOffsetArray(unittest.TestCase):
    def test_normalize(self):
        arr = np.ones((3, 3, 3), dtype='float32')
        chunk = OffsetArray(arr, (-1, -1, -1))
        mask = np.ones((3, 3, 3), dtype='float32')*0.5
        chunk.normalize(mask)
        self.assertTrue(np.all(chunk.array ==
                               np.ones((3, 3, 3), dtype='float32')*0.5))


if __name__ == '__main__':
    unittest.main()
