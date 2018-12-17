import numpy as np
import unittest
from chunkflow.block_inference_engine import BlockInferenceEngine  
from chunkflow.frameworks.identity_patch_inference_engine import IdentityPatchInferenceEngine
from chunkflow.offset_array import OffsetArray 

class TestBlockInferenceEngine(unittest.TestCase):
    def test_block_inference(self):
        patch_engine = IdentityPatchInferenceEngine()
        block_inference_engine = BlockInferenceEngine(
            patch_inference_engine=patch_engine, 
            patch_size = (32,256,256),
            patch_overlap=(4,64,64),
            num_output_channels=1,
        )
        
        image = np.random.rand(28*2+4, (256-64)*2+64, (256-64)*2+64)
        image = np.asarray(image, dtype='float32')
        image = OffsetArray(image) 
        output = block_inference_engine(image) 
        output = np.reshape(output, image.shape) 
        
        # some of the image voxel is 0, the test can only work with rtol=1 
        np.testing.assert_allclose(image, output, rtol=1, atol=1e-17) 


if __name__ == '__main__':
    unittest.main()
