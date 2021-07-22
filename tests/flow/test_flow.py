import numpy as np

from cloudvolume import CloudVolume
#from cloudvolume.volumecutout import VolumeCutout
from cloudvolume.lib import generate_random_string, Bbox
import os, shutil

from chunkflow.chunk.image.convnet.inferencer import Inferencer
from chunkflow.flow.flow import *


def test_inference_pipeline():
    # compute parameters
    mip = 0
    input_size = (36, 448, 448)
    patch_size = (20, 256, 256)
    cropping_margin_size = (4, 64, 64)
    patch_overlap = (4, 64, 64)
    input_mask_mip = 1
    input_mask_size = (
        input_size[0],
        input_size[1] // (2**input_mask_mip),
        input_size[2] // (2**input_mask_mip),
    )
    output_size = tuple(
        i - 2 * c
        for i, c in zip(input_size, cropping_margin_size))
    output_mask_mip = 2
    output_mask_size = (
        output_size[0],
        output_size[1] // (2**output_mask_mip),
        output_size[2] // (2**output_mask_mip),
    )
    output_bbox = Bbox.from_slices(
        tuple(slice(c, i - c)
            for i, c in zip(input_size, cropping_margin_size)))

    # create image dataset using cloud-volume
    img = np.random.randint(0, 256, size=input_size)
    img = img.astype(np.uint8)
    # save the input to disk
    input_volume_path = 'file:///tmp/input/' + generate_random_string(
    )
    CloudVolume.from_numpy(np.transpose(img),
                            vol_path=input_volume_path)

    # create input mask volume
    input_mask = np.ones(input_size, dtype=bool)
    input_mask_volume_path = 'file:///tmp/input-mask/' + generate_random_string()
    # the max_mip in CloudVolume is actually mip_stop 
    # it is not inclusive, so we need to +1 here
    CloudVolume.from_numpy(np.transpose(input_mask),
                            vol_path=input_mask_volume_path,
                            max_mip=input_mask_mip+1)
    input_mask = np.ones(input_mask_size, dtype=bool)
    # will mask out the [:2, :8, :8] since it is in mip 1
    input_mask[:(4 + 2), :(64 // 2 + 8 // 2), :(64 // 2 + 8 // 2)] = False
    input_mask_vol = CloudVolume(input_mask_volume_path,
                                    mip=input_mask_mip)
    input_mask_vol[:, :, :] = np.transpose(input_mask)

    # create output layer
    out = np.random.rand(3, *output_size).astype(np.float32)
    output_volume_path = 'file:///tmp/output/' + generate_random_string()
    output_vol = CloudVolume.from_numpy(
        np.transpose(out),
        vol_path=output_volume_path,
        voxel_offset=cropping_margin_size[::-1])

    # create output mask volume
    # this is the mip 0 size, so the size should be the same with output
    # it was only used to create the volume
    # TODO: delete this step by creating a mip parameter in from_numpy function
    output_mask = np.ones(output_size, dtype=bool)
    output_mask_volume_path = 'file:///tmp/output-mask/' + generate_random_string(
    )
    CloudVolume.from_numpy(np.transpose(output_mask),
                            vol_path=output_mask_volume_path,
                            max_mip=output_mask_mip + 1,
                            voxel_offset=cropping_margin_size[::-1])
    # this is the higher mip level mask, so this time we are using the real size
    output_mask = np.ones(output_mask_size, dtype=bool)
    # will mask out the [-2:, -8:, -8:] since it is in mip 2
    output_mask[-2:, -8 // 4:, -8 // 4:] = False
    output_mask_vol = CloudVolume(output_mask_volume_path,
                                    mip=output_mask_mip)
    output_mask_vol[:, :, :] = np.transpose(output_mask)

    # create volume for output thumbnail
    thumbnail_volume_path = os.path.join(output_volume_path,
                                                'thumbnail')
    thumbnail = np.asarray(out, dtype='uint8')
    CloudVolume.from_numpy(np.transpose(thumbnail),
                            vol_path=thumbnail_volume_path,
                            voxel_offset=cropping_margin_size[::-1],
                            max_mip=4)

    # run pipeline by composing functions
    print('cutout image chunk...')
    cutout_operator = ReadPrecomputedOperator(
        input_volume_path,
        mip=mip,
        expand_margin_size=cropping_margin_size)
    chunk = cutout_operator(output_bbox)

    print('mask input...')
    mask_input_operator = MaskOperator(input_mask_volume_path,
                                        input_mask_mip,
                                        mip,
                                        inverse=False)
    chunk = mask_input_operator(chunk)

    print('run convnet inference...')
    with Inferencer(None, None, patch_size,
                    num_output_channels=3,
                    input_size=chunk.shape,
                    output_patch_overlap=patch_overlap,
                    framework='identity',
                    batch_size=5,
                    dtype='float32') as inferencer:
        print(inferencer.compute_device)
        chunk = inferencer(chunk)
    print('after inference: {}'.format(chunk.slices))
    print('crop the marging...')
    chunk = chunk.crop_margin(output_bbox=output_bbox)
    print('after crop: {}'.format(chunk.slices))

    print('mask the output...')
    mask_output_operator = MaskOperator(output_mask_volume_path,
                                        output_mask_mip,
                                        mip,
                                        inverse=False)
    chunk = mask_output_operator(chunk)
    print('after masking: {}'.format(chunk.slices))

    print('save to output volume...')
    save_operator = WritePrecomputedOperator(output_volume_path,
                                    mip,
                                    upload_log=True,
                                    create_thumbnail=True)
    save_operator(chunk, log={'timer': {'save': 34}})
    print('after saving: {}'.format(chunk.slices))

    # evaluate the output
    print('start evaluation...')
    out = output_vol[output_bbox.to_slices()[::-1] +
                            (slice(0, 3), )]
    out = np.asarray(out)
    out = out[:, :, :, 0] * 255
    out = out.astype(np.uint8)
    out = np.transpose(out)

    # ignore the patch overlap around the border
    img = img[4:-4, 64:-64, 64:-64]

    # check that the masked region are all zero
    # input mask validation
    np.testing.assert_array_equal(out[:2, :8, :8], 0)
    # output mask validation
    np.testing.assert_array_equal(out[-2:, -8:, -8:], 0)

    # ignore the masked part of output
    img = img[2:-2, 8:-8, 8:-8]
    out = out[2:-2, 8:-8, 8:-8]

    # the value can only be close since there is mask error
    # abs(desired-actual) < 1.5 * 10**(-decimal)
    np.testing.assert_array_almost_equal(img, out, decimal=0)

    # clean up
    shutil.rmtree('/tmp/input')
    shutil.rmtree('/tmp/input-mask')
    shutil.rmtree('/tmp/output-mask')
    shutil.rmtree('/tmp/output')
