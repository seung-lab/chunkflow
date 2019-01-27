import unittest 
from cloudvolume import CloudVolume
import numpy as np
from cloudvolume.volumecutout import VolumeCutout
from cloudvolume.lib import generate_random_string, Bbox
from chunkflow.executor import Executor
import shutil


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.image_size = (36, 448, 448)
        # create dataset using cloud-volume 
        img = np.random.randint(0, 255+1, size=self.image_size)
        self.img = np.asarray(img, dtype='uint8') 
        
        # create temporal directories and info file
        # save the image to disk 
        self.image_layer_path = 'file://tmp/' + generate_random_string()
        CloudVolume.from_numpy(self.img, vol_path=self.image_layer_path)

        # create output layer 
        self.output_size = (28, 320, 320)
        out = np.random.rand(*self.output_size)
        out = np.asarray(out, dtype=np.float32)
        self.output_layer_path = 'file://tmp/' + generate_random_string()
        self.output_vol = CloudVolume.from_numpy(out, vol_path=self.output_layer_path) 

    def test_executor(self):
        executor = Executor(self.image_layer_path, self.output_layer_path, 
                            '', '', None, None, (20,256,256), (4,64,64), (4,64,64), 
                            framework='identity', mip=0)
        output_bbox = Bbox.from_list([64,64,4, *self.output_size][::-1])
        executor(output_bbox)
        out = self.output_vol[:,:,:]
        out = np.asarray(out, dtype=np.uint8) 
        assert np.alltrue(out == self.img[4:-4, 64:-64, 64:-64])
        shutil.rmtree('tmp')

if __name__ == '__main__':
    unittest.main()