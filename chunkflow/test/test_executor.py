import unittest 
from cloudvolume import CloudVolume
import numpy as np
from cloudvolume.volumecutout import VolumeCutout
from cloudvolume.lib import generate_random_string, Bbox
from chunkflow.executor import Executor
import os, shutil


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.image_size = (36, 448, 448)
        # create dataset using cloud-volume 
        img = np.random.randint(0, 255+1, size=self.image_size)
        self.img = img.astype(np.uint8) 
        # save the image to disk 
        self.image_layer_path = 'file:///tmp/image/' + generate_random_string()
        CloudVolume.from_numpy(np.transpose(self.img), vol_path=self.image_layer_path)
        
        # create image mask volume
        image_mask = np.ones((36, 448, 448), dtype=np.bool)
        self.image_mask_layer_path = 'file:///tmp/image-mask/' + generate_random_string()
        self.image_mask_mip = 1
        CloudVolume.from_numpy(np.transpose(image_mask), 
                               vol_path=self.image_mask_layer_path,
                               mip_num=self.image_mask_mip+1)
        image_mask = np.ones((36, 224, 224), dtype=np.bool)
        image_mask[8:-8, 120:-120, 120:-120] = True 
        image_mask_vol = CloudVolume(self.image_mask_layer_path, mip=self.image_mask_mip)
        image_mask_vol[:,:,:] = np.transpose(image_mask) 

        # create output layer 
        self.output_size = (28, 320, 320)
        out = np.random.rand(*self.output_size)
        out = np.asarray(out, dtype=np.float32)

        self.output_layer_path = 'file:///tmp/affinitymap/' + generate_random_string()
        self.output_vol = CloudVolume.from_numpy(np.transpose(out), vol_path=self.output_layer_path,
                                                 voxel_offset=(64,64,4))

        # create volume for output thumbnail
        thumbnail_path = os.path.join(self.output_layer_path, 'thumbnail')
        thumbnail = np.asarray(out, dtype='uint8')
        output_thumbnail_vol = CloudVolume.from_numpy(thumbnail, vol_path=thumbnail_path,
                                                      voxel_offset=(64,64,4), mip_num=5)


    def test_executor(self):
        executor = Executor(self.image_layer_path, self.output_layer_path, 
                            None, None, (20,256,256), (4,64,64), (4,64,64),
                            image_mask_layer_path=self.image_mask_layer_path, 
                            #output_mask_layer_path=self.output_mask_layer_path,
                            image_mask_mip=1, output_mask_mip=1,
                            framework='identity', mip=0, num_output_channels=1)
        output_size = np.asarray(self.output_size)
        output_start = np.asarray((4,64,64))
        output_stop = output_start + output_size
        output_bbox = Bbox.from_list([*output_start, *output_stop])
        executor(output_bbox)
        out = self.output_vol[64:448-64, 64:448-64, 4:32]
        out = np.transpose(out)
        out = out * 255
        out = np.ascontiguousarray(out, dtype=np.uint8)
        out[0,:,:,:]
        
        img = np.ascontiguousarray(self.img)
        # mask out the region
        img[16:-16, 240:-240, 240:-240] = 0
        img = img[4:-4, 64:-64, 64:-64]
        
        # the value can only be close since there is mask error
        assert np.alltrue(np.isclose(img, out, atol=1))
        shutil.rmtree('/tmp/image')
        shutil.rmtree('/tmp/image-mask')
        shutil.rmtree('/tmp/affinitymap')


if __name__ == '__main__':
    unittest.main()
