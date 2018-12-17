import numpy as np
from .patch_inference_engine import PatchInferenceEngine 

class IdentityPatchInferenceEngine(PatchInferenceEngine):
    """
        IdentityPatchInferenceEngine(PatchInferenceEngine) 

    return the same output with the input 
    this class was only used for tests 
    """
    def __init__(self):
        super().__init__()

    def __call__(self, patch):
        """
        return the same with argument 
        reshape the size to 5 dimension:
        batch, channel, z, y, x 
        """
        return np.reshape(patch, (1, 1, *patch.shape)) 




