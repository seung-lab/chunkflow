import numpy as np

from chunkflow.flow.inference import InferenceOperator
from chunkflow.flow.create_chunk import CreateChunkOperator


def test_inference():
    # compute parameters
    input_size = (18, 224, 224)
    patch_overlap = (2, 32, 32)
    patch_size = (10, 128, 128)

    image = CreateChunkOperator()(size=input_size, dtype='uint8')
    inference_operator = InferenceOperator(
        None,
        None,
        patch_size=patch_size,
        output_key='affinity',
        num_output_channels=3,
        patch_overlap=patch_overlap,
        framework='identity')
    output = inference_operator(image)

    # ignore the cropping region
    output = output[0, 2:-2, 32:-32, 32:-32]
    image = image[2:-2, 32:-32, 32:-32]

    output = output * 255
    output = output.astype(np.uint8)

    assert np.alltrue(np.isclose(image, output, atol=1))
