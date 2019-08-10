import unittest
import numpy as np

from chunkflow.flow.inference import InferenceOperator
from chunkflow.flow.create_chunk import CreateChunkOperator


class TestInference(unittest.TestCase):
    def setUp(self):
        # compute parameters
        self.mip = 0
        self.input_size = (36, 448, 448)
        self.cropping_margin_size = (4, 64, 64)
        self.patch_overlap = (4, 64, 64)
        self.input_mask_mip = 1

    def test(self):
        image = CreateChunkOperator()(size=self.input_size, dtype='uint8')
        inference_operator = InferenceOperator(
            None,
            None,
            patch_size=(20, 256, 256),
            output_key='affinity',
            num_output_channels=3,
            patch_overlap=self.patch_overlap,
            framework='identity')
        output = inference_operator(image)

        # ignore the cropping region
        output = output[0, 4:-4, 64:-64, 64:-64]
        image = image[4:-4, 64:-64, 64:-64]

        output = output * 255
        output = output.astype(np.uint8)

        self.assertTrue(np.alltrue(np.isclose(image, output, atol=1)))
