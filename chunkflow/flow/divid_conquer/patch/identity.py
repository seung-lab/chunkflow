import platform
import numpy as np
from .base import PatchInferencerBase


class Identity(PatchInferencerBase):
    """
        IdentityPatchInferenceEngine(PatchInferenceEngine) 

    return the same output with the input 
    this class was only used for tests 
    Note that the bump parameter is not used now, only support bump function
    """
    def __init__(self, convnet_model: str, convnet_weight_path: str,
                 input_patch_size: tuple, output_patch_overlap: tuple,
                 output_patch_size: tuple = None, 
                 num_output_channels: int = 1, 
                 dtype='float32',
                 bump: str='wu'):
        assert bump == 'wu'
        super().__init__(input_patch_size, output_patch_size,
                         output_patch_overlap, num_output_channels,
                         dtype=dtype)
    
    @property
    def compute_device(self):
        return platform.processor()

    def __call__(self, input_patch):
        """
        return the same with argument
        reshape the size to 5 dimension:
        batch, channel, z, y, x
        """
        input_patch = self._reshape_patch_to_5d(input_patch)

        output_patch = input_patch.astype(np.float32)
        #if np.issubdtype(patch.dtype, np.integer):
        #    # normalize to 0-1 value range
        #    output /= np.iinfo(patch.dtype).max

        output_patch = self._crop_output_patch(output_patch)
        
        # mask should be done in patch engine now
        output_patch *= self.output_patch_mask_numpy
        
        if self.num_output_channels > 1:
            output_patch = np.repeat(output_patch, 
                                     self.num_output_channels, axis=1)

        return output_patch
