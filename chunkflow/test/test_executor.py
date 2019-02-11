import unittest 
from cloudvolume import CloudVolume
import numpy as np
from cloudvolume.volumecutout import VolumeCutout
from cloudvolume.lib import generate_random_string, Bbox
from chunkflow.executor import Executor
import os, shutil


class TestExecutor(unittest.TestCase):
    def setUp(self):
        # compute parameters 
        self.image_size = (36, 448, 448)
        self.cropping_margin_size = (4, 64, 64)
        self.patch_overlap = (4, 64, 64)
        self.image_mask_mip = 1
        self.image_mask_size = (self.image_size[0], 
                                self.image_size[1]//(2**self.image_mask_mip), 
                                self.image_size[2]//(2**self.image_mask_mip),)
        self.output_size = tuple(i-2*c for i,c in zip(self.image_size, 
                                                      self.cropping_margin_size))
        self.output_mask_mip = 2 
        self.output_mask_size = (self.output_size[0],
                                 self.output_size[1]//(2**self.output_mask_mip),
                                 self.output_size[2]//(2**self.output_mask_mip),)

        # create dataset using cloud-volume 
        img = np.random.randint(0, 255+1, size=self.image_size)
        self.img = img.astype(np.uint8) 
        # save the image to disk 
        self.image_layer_path = 'file:///tmp/image/' + generate_random_string()
        CloudVolume.from_numpy(np.transpose(self.img), vol_path=self.image_layer_path)
        
        # create image mask volume
        image_mask = np.ones(self.image_size, dtype=np.bool)
        self.image_mask_layer_path = 'file:///tmp/image-mask/' + generate_random_string()
        CloudVolume.from_numpy(np.transpose(image_mask), 
                               vol_path=self.image_mask_layer_path,
                               max_mip=self.image_mask_mip)
        image_mask = np.ones(self.image_mask_size, dtype=np.bool)
        # will mask out the [:2, :8, :8] since it is in mip 1
        image_mask[:(4+2), :(64//2+8//2), :(64//2+8//2)] = False 
        image_mask_vol = CloudVolume(self.image_mask_layer_path, mip=self.image_mask_mip)
        image_mask_vol[:,:,:] = np.transpose(image_mask) 

        # create output layer 
        out = np.random.rand(*self.output_size)
        out = np.asarray(out, dtype=np.float32)
        self.output_layer_path = 'file:///tmp/output/' + generate_random_string()
        self.output_vol = CloudVolume.from_numpy(np.transpose(out), 
                                                 vol_path=self.output_layer_path,
                                                 voxel_offset=self.cropping_margin_size[::-1])
        
        # create output mask volume
        # this is the mip 0 size, so the size should be the same with output
        # it was only used to create the volume
        # TODO: delete this step by creating a mip parameter in from_numpy function
        output_mask = np.ones(self.output_size, dtype=np.bool)
        self.output_mask_layer_path = 'file:///tmp/output-mask/' + generate_random_string()
        CloudVolume.from_numpy(np.transpose(output_mask), 
                               vol_path=self.output_mask_layer_path,
                               max_mip=self.output_mask_mip,
                               voxel_offset=self.cropping_margin_size[::-1])
        # this is the higher mip level mask, so this time we are using the real size
        output_mask = np.ones(self.output_mask_size, dtype=np.bool)
        # will mask out the [-2:, -8:, -8:] since it is in mip 2
        output_mask[-2:, -8//4:, -8//4:] = False 
        output_mask_vol = CloudVolume(self.output_mask_layer_path, mip=self.output_mask_mip)
        output_mask_vol[:,:,:] = np.transpose(output_mask) 

        # create volume for output thumbnail
        thumbnail_path = os.path.join(self.output_layer_path, 'thumbnail')
        thumbnail = np.asarray(out, dtype='uint8')
        output_thumbnail_vol = CloudVolume.from_numpy(thumbnail, vol_path=thumbnail_path,
                                                      voxel_offset=(64,64,4), max_mip=4)

    def test_executor(self):
        executor = Executor(self.image_layer_path, self.output_layer_path, 
                            None, None, (20,256,256), (4,64,64), (4,64,64),
                            image_mask_layer_path=self.image_mask_layer_path, 
                            output_mask_layer_path=self.output_mask_layer_path,
                            image_mask_mip=self.image_mask_mip, 
                            output_mask_mip=self.output_mask_mip,
                            framework='identity', mip=0, num_output_channels=1)
        output_size = np.asarray(self.output_size)
        output_start = np.asarray((4,64,64))
        output_stop = output_start + output_size
        output_bbox = Bbox.from_list([*output_start, *output_stop])
        executor(output_bbox)
        out = self.output_vol[(slice(c, s-c) for c,s in 
                               zip(self.cropping_margin_size[::-1], self.image_size[::-1]))]
        out = np.transpose(out)
        out = out * 255
        out = np.ascontiguousarray(out, dtype=np.uint8)
        out = out[0,:,:,:]
        
        # ignore the patch overlap around the border
        img = self.img[4:-4, 64:-64, 64:-64]
        
        # check that the masked region are all zero
        # image mask validation
        assert np.alltrue(out[:2, :8, :8]==0)
        # output mask validation 
        assert np.alltrue(out[-2: -8:, -8:]==0)

        # ignore the masked part of output 
        img = img[2:-2, 8:-8, 8:-8]
        out = out[2:-2, 8:-8, 8:-8]
        # the value can only be close since there is mask error
        assert np.alltrue(np.isclose(img, out, atol=1))
        
        # clean up 
        shutil.rmtree('/tmp/image')
        shutil.rmtree('/tmp/image-mask')
        shutil.rmtree('/tmp/output-mask')
        shutil.rmtree('/tmp/output')


if __name__ == '__main__':
    unittest.main()
