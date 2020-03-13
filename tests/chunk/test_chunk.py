import numpy as np
import unittest
from cloudvolume.lib import Bbox
from chunkflow.chunk import Chunk


def create_chunk(size:tuple = (7, 8, 9), global_offset=(-2, -3, -4), 
                 dtype=np.float32):
    # make the tests consistent in multiple runs
    np.random.seed(1)
    arr = np.random.rand(*size).astype(dtype)
    chunk = Chunk(arr, global_offset)
    return chunk

def test_channel_voting():
    chunk = create_chunk(size=(5, 2,3,4), 
                         global_offset=(0, 3,4,5), dtype=np.float32)
    out = chunk.channel_voting()
    
    gt = np.array([[[4, 1, 3, 2], [3, 1, 4, 3], [1, 3, 3, 4]], 
                   [[4, 4, 2, 3], [1, 1, 2, 3], [2, 0, 1, 0]]], np.uint8)
    # out latest voting result start from 1
    gt = gt + 1
    
    np.testing.assert_array_equal(out.array, gt)
    np.testing.assert_array_equal(out.global_offset, (3,4,5))

def test_mask_last_channel():
    size = (4, 3, 3, 3)
    global_offset = (0, -1, -1, -1)
    chunk = create_chunk(size=size, global_offset=global_offset)
    out = chunk.mask_using_last_channel()
    assert out.shape == (3,3,3,3)
    np.testing.assert_array_equal(out, chunk[:3, :,:,:])

def test_crop_margin():
    size = (7, 8, 9)
    global_offset = (-2, -3, -4)
    chunk = create_chunk(size=size, global_offset=global_offset)
    new_chunk = chunk.crop_margin(margin_size=(1, 2, 3))
    np.testing.assert_array_equal(new_chunk, chunk.array[1:-1, 2:-2, 3:-3])
    np.testing.assert_array_equal(
        new_chunk.global_offset,
        (-1, -1, -1)
    )


class Test3DChunk(unittest.TestCase):
    def setUp(self):
        self.size = (7, 8, 9)
        self.global_offset = (-1, -1, -1)
        arr = np.random.rand(*self.size).astype('float32')
        self.chunk = Chunk(arr, self.global_offset)

    #def test_math(self):
    #    self.assertEqual(np.max(self.chunk), np.max(self.chunk.array))
    #    self.assertEqual(np.min(self.chunk), np.min(self.chunk.array))

    def test_create_from_bounding_box(self):
        bbox = Bbox.from_delta(self.global_offset, self.size)
        Chunk.from_bbox( bbox )

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

        chunk2 = self.chunk.transpose()
        print('type of chunk after transpose: {}'.format(type(chunk2)))
        self.assertIsInstance(chunk2, Chunk)
        self.assertEqual(chunk2.shape, self.chunk.shape[::-1])
        self.assertEqual(chunk2.global_offset, self.chunk.global_offset[::-1])
        self.assertTrue( np.array_equal(chunk2.array, np.transpose(self.chunk.array)  ))

        #chunk2 = np.ascontiguousarray(chunk2)
        #self.assertIsInstance(chunk2, Chunk)
        #self.assertEqual(chunk2.global_offset, self.chunk.global_offset)

        #chunk2 = np.pad(chunk2, ((0,2),(0,2),(0,2)), 'reflect')
        #self.assertIsInstance(chunk2, Chunk)
        #self.assertEqual(chunk2.global_offset, self.chunk.global_offset)

    def test_normalize(self):
        print('\ntest chunk normalization...')
        arr = np.ones((3, 3, 3), dtype='float32')
        global_offset = (-1, -1, -1)
        chunk = Chunk(arr, global_offset=global_offset)

        mask = np.ones((3, 3, 3), dtype='float32') * 0.5
        mask = Chunk(mask, global_offset=global_offset)
        chunk *= mask
        self.assertTrue(chunk == mask)

    def test_item(self):
        arr = np.ones((1, 3, 3, 3), dtype='float32')
        chunk = Chunk(arr, (0, -1, -1, -1))
        chunk[:, :,:,:] = 0 
        self.assertTrue( chunk == 0 )

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
        selected2 = chunk.where(chunk.array > 0.5)
        for i1, i2 in zip(selected1, selected2):
            # print('i1: {}, i2: {}'.format(i1, i2))
            self.assertTrue((i1 - i2 == 1).all())


if __name__ == '__main__':
    unittest.main()
