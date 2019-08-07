import numpy as np
from .patch_mask import PatchMask


class PatchEngine(object):
    """PatchEngine
    
    the input patch is a 
    """

    def __init__(self, patch_size, patch_overlap):
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

        # prepare patch mask
        self.mask = PatchMask(patch_size, patch_overlap)

    def __call__(self, patch: np.ndarray) -> np.ndarray:
        r"""This method should be inherited for real implementation

        Args:
            patch: a image patch with datatype of float32,
                The value range should be in [0,1]
        
        Returns
        --------
        np.ndarray
        """
        return NotImplementedError()

    def _reshape_patch(self, patch):
        """patch should be a 5d np array
        """
        assert isinstance(patch, np.ndarray)
        if patch.ndim == 3:
            patch = patch.reshape((1, 1) + patch.shape)
        elif patch.ndim == 4:
            patch = patch.reshape((1, ) + patch.shape)
        return patch
