# from .inference_engine import InferenceEngine
# import imp
import numpy as np
from .base import PatchInferencerBase
from chunkflow.lib import load_source


class General(PatchInferencerBase):
    """perform inference for an image patch using an arbitrary backend.
    Parameters
    ----------
    patch_size: size of input/output patch size. We assume that 
        the input and output patch size is the same. 
    patch_overlap: overlap of neighboring patches.
    model_file_name: file name of model
    weight_file_name: file name of trained weight.
    num_output_channels: number of output channels.
    
    You can make some customized processing in your model file.
    You need to define a class called `PatchInferencer`.
    The constructor inputs are `model_weight_file`, `patch_mask`.
    You should define a `__call__` function to process the `input_patch`,
    and the output the `output_patch` after masking. 

    Because the masking can happen in the device, such as GPU or TPU, so 
    it is the PatchInferencer's job to perform the masking.
    """
    def __init__(self, convnet_model: str, convnet_weight_path: str,
                 input_patch_size: tuple, 
                 output_patch_size: tuple, 
                 output_patch_overlap: tuple,
                 num_output_channels: int = 1, 
                 dtype: str='float32',
                 bump: str='wu'):
        # To-Do: support zung function
        assert bump == 'wu'
        super().__init__(input_patch_size, output_patch_size, 
                         output_patch_overlap, num_output_channels, 
                         dtype=dtype)

        self.num_output_channels = num_output_channels
   
        net_source = load_source(convnet_model)

        assert hasattr(net_source, "PatchInferencer")
        self.patch_inferencer = net_source.PatchInferencer(
            convnet_weight_path, self.output_patch_mask)
    
    @property
    def compute_device(self):
        # To-Do: make the device available to log
        return 'unknown'

    def __call__(self, input_patch):
        # make sure that the patch is 5d ndarray
        input_patch = self._reshape_patch_to_5d(input_patch)
        output_patch = self.patch_inferencer( input_patch )
        assert isinstance(output_patch, np.ndarray)
        return output_patch
