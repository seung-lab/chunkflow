import unittest
import numpy as np

from cloudvolume import CloudVolume
#from cloudvolume.volumecutout import VolumeCutout
from cloudvolume.lib import generate_random_string, Bbox
import os, shutil

from chunkflow.chunk.image.convnet.inferencer import Inferencer
from chunkflow.flow.flow import *


class TestInferencePipeline(unittest.TestCase):
    def setUp(self):
        # compute parameters
        self.mip = 0
        self.input_size = (36, 448, 448)
        self.patch_size = (20, 256, 256)
        self.cropping_margin_size = (4, 64, 64)
        self.patch_overlap = (4, 64, 64)
        self.input_mask_mip = 1
        self.input_mask_size = (
            self.input_size[0],
            self.input_size[1] // (2**self.input_mask_mip),
            self.input_size[2] // (2**self.input_mask_mip),
        )
        self.output_size = tuple(
            i - 2 * c
            for i, c in zip(self.input_size, self.cropping_margin_size))
        self.output_mask_mip = 2
        self.output_mask_size = (
            self.output_size[0],
            self.output_size[1] // (2**self.output_mask_mip),
            self.output_size[2] // (2**self.output_mask_mip),
        )
        self.output_bbox = Bbox.from_slices(
            tuple(
                slice(c, i - c)
                for i, c in zip(self.input_size, self.cropping_margin_size)))
        #output_size = np.asarray(self.output_size)
        #output_start = np.asarray((4,64,64))
        #output_stop = output_start + output_size
        #output_bbox = Bbox.from_list([*output_start, *output_stop])

        # create image dataset using cloud-volume
        img = np.random.randint(0, 256, size=self.input_size)
        self.img = img.astype(np.uint8)
        # save the input to disk
        self.input_volume_path = 'file:///tmp/input/' + generate_random_string(
        )
        CloudVolume.from_numpy(np.transpose(self.img),
                               vol_path=self.input_volume_path)

        # create input mask volume
        input_mask = np.ones(self.input_size, dtype=np.bool)
        self.input_mask_volume_path = 'file:///tmp/input-mask/' + generate_random_string(
        )
        CloudVolume.from_numpy(np.transpose(input_mask),
                               vol_path=self.input_mask_volume_path,
                               max_mip=self.input_mask_mip)
        input_mask = np.ones(self.input_mask_size, dtype=np.bool)
        # will mask out the [:2, :8, :8] since it is in mip 1
        input_mask[:(4 + 2), :(64 // 2 + 8 // 2), :(64 // 2 + 8 // 2)] = False
        input_mask_vol = CloudVolume(self.input_mask_volume_path,
                                     mip=self.input_mask_mip)
        input_mask_vol[:, :, :] = np.transpose(input_mask)

        # create output layer
        out = np.random.rand(3, *self.output_size).astype(np.float32)
        self.output_volume_path = 'file:///tmp/output/' + generate_random_string(
        )
        self.output_vol = CloudVolume.from_numpy(
            np.transpose(out),
            vol_path=self.output_volume_path,
            voxel_offset=self.cropping_margin_size[::-1])

        # create output mask volume
        # this is the mip 0 size, so the size should be the same with output
        # it was only used to create the volume
        # TODO: delete this step by creating a mip parameter in from_numpy function
        output_mask = np.ones(self.output_size, dtype=np.bool)
        self.output_mask_volume_path = 'file:///tmp/output-mask/' + generate_random_string(
        )
        CloudVolume.from_numpy(np.transpose(output_mask),
                               vol_path=self.output_mask_volume_path,
                               max_mip=self.output_mask_mip,
                               voxel_offset=self.cropping_margin_size[::-1])
        # this is the higher mip level mask, so this time we are using the real size
        output_mask = np.ones(self.output_mask_size, dtype=np.bool)
        # will mask out the [-2:, -8:, -8:] since it is in mip 2
        output_mask[-2:, -8 // 4:, -8 // 4:] = False
        output_mask_vol = CloudVolume(self.output_mask_volume_path,
                                      mip=self.output_mask_mip)
        output_mask_vol[:, :, :] = np.transpose(output_mask)

        # create volume for output thumbnail
        self.thumbnail_volume_path = os.path.join(self.output_volume_path,
                                                  'thumbnail')
        thumbnail = np.asarray(out, dtype='uint8')
        CloudVolume.from_numpy(np.transpose(thumbnail),
                               vol_path=self.thumbnail_volume_path,
                               voxel_offset=self.cropping_margin_size[::-1],
                               max_mip=4)

    def test_inference_pipeline(self):
        # run pipeline by composing functions
        print('cutout image chunk...')
        cutout_operator = CutoutOperator(
            self.input_volume_path,
            mip=self.mip,
            expand_margin_size=self.cropping_margin_size)
        chunk = cutout_operator(self.output_bbox)

        print('mask input...')
        mask_input_operator = MaskOperator(self.input_mask_volume_path,
                                           self.input_mask_mip,
                                           self.mip,
                                           inverse=False)
        chunk = mask_input_operator(chunk)

        print('run convnet inference...')
        with Inferencer(None, None, self.patch_size,
                        num_output_channels=3,
                        input_size=chunk.shape,
                        output_patch_overlap=self.patch_overlap,
                        framework='identity',
                        batch_size=5,
                        dtype='float32') as inferencer:
            print(inferencer.compute_device)
            chunk = inferencer(chunk)
        print('after inference: {}'.format(chunk.slices))

        print('crop the marging...')
        chunk = chunk.crop_margin(output_bbox=self.output_bbox)
        print('after crop: {}'.format(chunk.slices))

        print('mask the output...')
        mask_output_operator = MaskOperator(self.output_mask_volume_path,
                                            self.output_mask_mip,
                                            self.mip,
                                            inverse=False)
        chunk = mask_output_operator(chunk)
        print('after masking: {}'.format(chunk.slices))

        print('save to output volume...')
        save_operator = SaveOperator(self.output_volume_path,
                                     self.mip,
                                     upload_log=True,
                                     create_thumbnail=True)
        save_operator(chunk, log={'timer': {'save': 34}})
        print('after saving: {}'.format(chunk.slices))

        # evaluate the output
        print('start evaluation...')
        out = self.output_vol[self.output_bbox.to_slices()[::-1] +
                              (slice(0, 3), )]
        out = np.asarray(out)
        out = out[:, :, :, 0] * 255
        out = out.astype(np.uint8)
        out = np.transpose(out)

        # ignore the patch overlap around the border
        img = self.img[4:-4, 64:-64, 64:-64]

        # check that the masked region are all zero
        # input mask validation
        self.assertTrue(np.alltrue(out[:2, :8, :8] == 0))
        # output mask validation
        self.assertTrue(np.alltrue(out[-2:-8:, -8:] == 0))

        # ignore the masked part of output
        img = img[2:-2, 8:-8, 8:-8]
        out = out[2:-2, 8:-8, 8:-8]

        # the value can only be close since there is mask error
        self.assertTrue(np.alltrue(np.isclose(img, out, atol=1)))

        # clean up
        shutil.rmtree('/tmp/input')
        shutil.rmtree('/tmp/input-mask')
        shutil.rmtree('/tmp/output-mask')
        shutil.rmtree('/tmp/output')


if __name__ == '__main__':
    unittest.main()
