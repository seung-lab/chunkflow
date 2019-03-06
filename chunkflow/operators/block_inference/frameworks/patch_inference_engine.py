import numpy as np

class PatchInferenceEngine(object):
    def __init__(self):
        pass

    def __call__(self, patch):
        """
        this method should be inherited for real implementation
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
