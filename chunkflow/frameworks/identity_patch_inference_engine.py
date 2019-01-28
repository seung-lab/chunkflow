import numpy as np
from .patch_inference_engine import PatchInferenceEngine


class IdentityPatchInferenceEngine(PatchInferenceEngine):
    """
        IdentityPatchInferenceEngine(PatchInferenceEngine) 

    return the same output with the input 
    this class was only used for tests 
    """

    def __init__(self, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels

    def __call__(self, patch):
        """
        return the same with argument 
        reshape the size to 5 dimension:
        batch, channel, z, y, x 
        """
        output = np.reshape(patch, (1, 1, *patch.shape))
        if self.num_output_channels > 1:
            output = np.repeat(output, 3, axis=1)
        
        return output
