import numpy as np
from chunkflow.chunk.image.convnet.inferencer import Inferencer
from chunkflow.chunk.image.convnet.patch.identity import Identity
from chunkflow.chunk import Chunk


def test_aligned_input_size():
    print('\ntest block inference with aligned input size...')
    # compute parameters
    input_size = (18, 224, 224)
    patch_overlap = (2, 32, 32)
    input_patch_size = (10, 128, 128)

    image = Chunk.create(size=input_size, dtype='uint8')
    with Inferencer(None, None,
                    input_patch_size,
                    num_output_channels=3,
                    output_patch_overlap=patch_overlap,
                    input_size=input_size,
                    framework='identity', dtype='float32') as inferencer:
        output = inferencer(image)
    
    # ignore the cropping region
    output = output[0, :, :, :]
    image = image[2:-2, 32:-32, 32:-32]

    output = output * 255
    output = output.astype(np.uint8)
    
    assert np.alltrue(np.isclose(image, output, atol=1))

def test_aligned_patch_num():
    print('\ntest block inference with patch number...')
    patch_size = (32, 256, 256)
    patch_overlap = (4, 64, 64)
    patch_num = (2, 2, 2)
    num_output_channels = 2
    dtype = 'float16'
    
    image = np.random.randint(
        1, 255, 
        size=(28 * 2 + 4, (256 - 64) * 2 + 64, (256 - 64) * 2 + 64),
        dtype=np.uint8)
    image = Chunk(image)

    with Inferencer(None, None, patch_size,
                    output_patch_overlap=patch_overlap,
                    num_output_channels=num_output_channels,
                    patch_num = patch_num,
                    framework='identity',
                    dtype=dtype,
                    batch_size=5,
                    mask_output_chunk=False) as block_inference_engine:
        output = block_inference_engine(image)

    # only use the first channel to check correctness
    output = output[0, :, :, :]

    # we need to crop the patch overlap since the values were changed
    image = image[patch_overlap[0]:-patch_overlap[0], patch_overlap[1]:
                  -patch_overlap[1], patch_overlap[2]:-patch_overlap[2]]

    image = image.astype(dtype) / 255
    print('maximum difference: ', np.max(image - output))

    # some of the image voxel is 0, the test can only work with rtol=1
    np.testing.assert_allclose(image, output, rtol=1e-3, atol=1e-3)


def test_aligned_input_chunk_with_croped_patch():
    print('\ntest block inference engine...')
    input_patch_size = (20, 256, 256)
    output_patch_size = (16, 192, 192)
    output_patch_crop_margin = (2, 32, 32)
    output_patch_overlap = (2, 32, 32)
    input_patch_overlap = (6, 96, 96)
    input_patch_stride = (14, 160, 160)
    input_size = (2*14+6, 2*160+96, 2*160+96)
    num_output_channels = 1
     
    image = np.random.randint(
        1, 255,
        size=input_size, 
        dtype=np.uint8)
    
    # make sure that it works with arbitrary global offset
    image = Chunk(image, global_offset=(123, 345, 567))
   
    with Inferencer(None, None, input_patch_size,
                    output_patch_size = output_patch_size,
                    output_patch_overlap=output_patch_overlap,
                    patch_num = (2, 2, 2),
                    num_output_channels=num_output_channels,
                    framework='identity',
                    batch_size=5,
                    mask_output_chunk=False) as chunk_inferencer:
        output = chunk_inferencer(image)

    # only use the first channel to check correctness
    output = output[0, :, :, :]
    assert output.ndim == 3
    
    # we need to crop the patch overlap since the values were changed
    image = image[4:-4, 64:-64, 64:-64]
    image = image.astype(np.float32) / 255
     
    print('maximum difference: ', np.max(image - output))
    
    # some of the image voxel is 0, the test can only work with rtol=1
    np.testing.assert_allclose(image, output, rtol=1e-5, atol=1e-5)

def test_non_aligned_input_chunk():
    print('\ntest no aligned block inference engine...')
    patch_size = (32, 256, 256)
    patch_overlap = (4, 64, 64)
    num_output_channels = 2
    input_size = (28 * 2 + 4 + 6, (256 - 64) * 2 + 64 + 7,
                                    (256 - 64) * 2 + 64 + 9) 
    image = np.random.randint(1, 255, size=input_size,
                              dtype=np.uint8)
    image = Chunk(image)

    with Inferencer(None, None, patch_size,
                    output_patch_overlap=patch_overlap,
                    num_output_channels=num_output_channels,
                    batch_size=5,
                    framework='identity',
                    mask_output_chunk=True) as inferencer:
        output = inferencer(image)
    
    # only use the first channel to check correctness
    output = output[0, :, :, :]
    #output = np.reshape(output, image.shape)

    image = image.astype(np.float32) / 255
    residual = image - output 
    print('maximum difference: ', np.max(residual.array))

    # some of the image voxel is 0, the test can only work with rtol=1
    np.testing.assert_allclose(image, output, rtol=1e-5, atol=1e-5)
