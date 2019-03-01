import unittest
import numpy as np
import os

from chunkflow.lib.offset_array import OffsetArray

from .read_file import ReadFileOperator
from .write_h5 import WriteH5Operator


class TestReadWrite(unittest.TestCase):
    def test_read_write(self):
        arr = np.random.randint(0, 256, size=(8, 16, 16), 
                                dtype=np.uint8)
        chunk = OffsetArray(arr, global_offset=(1,2,3))
        
        file_name = 'test.h5'
        if os.path.exists(file_name):
            os.remove(file_name)
        
        WriteH5Operator()(chunk, file_name)
        chunk2 = ReadFileOperator()( file_name )
        self.assertTrue( np.alltrue(chunk==chunk2) )
        self.assertEqual(chunk.global_offset, 
                         chunk2.global_offset, 
                         msg='the global offset is different.')
        os.remove( file_name )


if __name__ == '__main__':
    unittest.main()
