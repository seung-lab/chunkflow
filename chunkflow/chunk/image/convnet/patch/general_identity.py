import numpy as np


class PatchInferencer:
    """
        IdentityPatchInferenceEngine(PatchInferenceEngine) 

    return the same output with the input 
    this class was only used for tests 
    Note that the bump parameter is not used now, only support bump function
    """
    def __init__(self, model_weight_file, output_patch_mask):
        self.output_patch_mask = output_patch_mask

    def __call__(self, input_patch):
        """
        return the same with argument
        reshape the size to 5 dimension:
        batch, channel, z, y, x
        """
        if np.issubdtype(input_patch.dtype, np.integer):
            # normalize to 0-1 value range
            input_patch /= np.iinfo(input_patch.dtype).max
        output_patch = input_patch.astype(np.float32)

        # mask should be done in patch engine now
        output_patch *= self.output_patch_mask
        
        return output_patch
