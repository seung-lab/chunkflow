import numpy as np
import unittest
from .block_inference_engine import BlockInferenceEngine
from .frameworks.identity_patch_inference_engine import IdentityPatchInferenceEngine
from chunkflow.chunk import Chunk


class TestBlockInferenceEngine(unittest.TestCase):
    def test_aligned_input_chunk(self):
        print('\ntest block inference engine...')
        patch_overlap = (4, 64, 64)
        patch_engine = IdentityPatchInferenceEngine()
        block_inference_engine = BlockInferenceEngine(
            patch_inference_engine=patch_engine,
            patch_size=(32, 256, 256),
            patch_overlap=patch_overlap,
            num_output_channels=1,
            batch_size=2
        )
        
        image = np.random.randint(0, 255, size=(28 * 2 + 4, 
                                                (256 - 64) * 2 + 64,
                                                (256 - 64) * 2 + 64), 
                                  dtype=np.uint8) 
        image = Chunk(image)
        output = block_inference_engine(image)
        output = np.reshape(output, image.shape)
        
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
            image /= 255

        # we need to crop the patch overlap since the values were changed
        image = image[patch_overlap[0]:-patch_overlap[0],
                      patch_overlap[1]:-patch_overlap[1],
                      patch_overlap[2]:-patch_overlap[2]]
        output = output[patch_overlap[0]:-patch_overlap[0],
                        patch_overlap[1]:-patch_overlap[1],
                        patch_overlap[2]:-patch_overlap[2]]
        # some of the image voxel is 0, the test can only work with rtol=1
        np.testing.assert_allclose(image, output, rtol=1, atol=1e-17)
    

if __name__ == '__main__':
    # setup debug mode
    # https://docs.python.org/3.5/library/asyncio-dev.html#debug-mode-of-asyncio
    #import os
    #os.environ['PYTHONASYNCIODEBUG'] = '1'
    #import logging
    #logging.basicConfig(level=logging.DEBUG)
    unittest.main()
