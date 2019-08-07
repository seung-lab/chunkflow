import numpy as np
from .base import PatchEngine


class Identity(PatchEngine):
    """
        IdentityPatchInferenceEngine(PatchInferenceEngine) 

    return the same output with the input 
    this class was only used for tests 
    """

    def __init__(self, patch_size, patch_overlap, num_output_channels=1):
        super().__init__(patch_size, patch_overlap)
        self.num_output_channels = num_output_channels

    def __call__(self, patch):
        """
        return the same with argument
        reshape the size to 5 dimension:
        batch, channel, z, y, x
        """
        patch = self._reshape_patch(patch)

        output = patch.astype(np.float32)
        #if np.issubdtype(patch.dtype, np.integer):
        #    # normalize to 0-1 value range
        #    output /= np.iinfo(patch.dtype).max

        # mask should be done in patch engine now
        output *= self.mask

        if self.num_output_channels > 1:
            output = np.repeat(output, self.num_output_channels, axis=1)

        return output
