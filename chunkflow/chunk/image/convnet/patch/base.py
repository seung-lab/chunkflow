import numpy as np
from .patch_mask import PatchMask


class PatchInferencerBase(object):
    """PatchEngine
    
    the input patch is a 
    """
    def __init__(self, input_patch_size: tuple, output_patch_size: tuple, 
                 output_patch_overlap: tuple, num_output_channels: int,
                 dtype: str='float32'):
        
        if output_patch_size is None:
            output_patch_size = input_patch_size
        
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        self.output_patch_overlap = output_patch_overlap
        self.num_output_channels = num_output_channels
        
        assert len(output_patch_overlap) == 3
        assert len(input_patch_size) == 3
        assert len(output_patch_size) == 3

        self.output_offset = tuple((osz-isz)//2 for osz, isz in 
                                                   zip(input_patch_size, output_patch_size))
        
        self.input_patch_overlap = tuple((opo + 2 * ocms) for opo, ocms in 
                                         zip(output_patch_overlap, 
                                             self.output_offset))

        self.input_patch_stride = tuple(p - o for p, o in
                                        zip(input_patch_size, self.input_patch_overlap))
        self.output_patch_stride = tuple(p - o for p, o in 
                                         zip(output_patch_size, self.output_patch_overlap))

        # prepare patch mask
        self.output_patch_mask = PatchMask(output_patch_size, 
                                           output_patch_overlap,
                                           dtype=dtype)
        # keep a version in cpu for making chunk mask
        self.output_patch_mask_numpy = self.output_patch_mask

    def __call__(self, input_patch: np.ndarray) -> np.ndarray:
        r"""This method should be inherited for real implementation

        Args:
            patch: a image patch with datatype of float32,
                The value range should be in [0,1]
        
        Returns
        --------
        np.ndarray
        """
        return NotImplementedError('this function should be overload by inherited class!')

    def _reshape_patch_to_5d(self, input_patch):
        """patch should be a 5d np array
        """
        assert isinstance(input_patch, np.ndarray)
        if input_patch.ndim == 3:
            input_patch = input_patch.reshape((1, 1) + input_patch.shape)
        elif input_patch.ndim == 4:
            input_patch = input_patch.reshape((1, ) + input_patch.shape)
        return input_patch

    def _crop_output_patch(self, output_patch):
        return output_patch[:, :self.num_output_channels,
                            self.output_offset[0]:output_patch.shape[-3]-self.output_offset[0],
                            self.output_offset[1]:output_patch.shape[-2]-self.output_offset[1],
                            self.output_offset[2]:output_patch.shape[-1]-self.output_offset[2]]

