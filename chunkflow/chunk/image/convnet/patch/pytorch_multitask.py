import os
import torch
from pytorch_model.model import load_model
from types import SimpleNamespace

from .base import PatchInferencerBase

class PyTorchMultitask(PatchInferencerBase):
    """
        this function do masking in gpu for speed up, 
        so we need the patch_overlap information.
    """

    def __init__(self, convnet_model: str, convnet_weight_path: str,
                 input_patch_size: tuple, output_patch_overlap: tuple,
                 output_patch_size: tuple = None, 
                 num_output_channels: int = 1, bump: str='wu'):
        
        super().__init__(input_patch_size, output_patch_size,
                         output_patch_overlap, num_output_channels)

        # we currently only support two types of model
        assert convnet_model in ('rsunet', 'rsunet_act')

        self.output_key = 'affinity'

        width = (16, 32, 64, 128)
        original_num_output_channels = 3

        d = {
            'model': convnet_model,
            'width': width,
            'in_spec': {
                'input': (1, *input_patch_size)
            },
            'out_spec': {
                self.output_key: (original_num_output_channels, *self.output_patch_size)
            },
            'scan_spec': {
                self.output_key: (num_output_channels, *self.output_patch_size)
            },
            'cropsz': self.output_patch_size,
            'pretrain': True,
            #'precomputed': torch.cuda.is_available(),
            'precomputed': False,
            'edges': [(0, 0, 1), (0, 1, 0), (1, 0, 0)],
            'overlap': self.output_patch_overlap,
            'bump': bump
        }

        self.opt = SimpleNamespace(**d)
        assert os.path.isfile(convnet_weight_path)
        self.net = load_model(self.opt, convnet_weight_path)
        breakpoint()
        assert len(self.opt.in_spec) == 1
        
        if torch.cuda.is_available():
            # put mask to gpu
            self.output_patch_mask = torch.from_numpy(
                            self.output_patch_mask).cuda()
 
    def __call__(self, input_patch):
        # make sure that patch is 5d ndarray
        input_patch = self._reshape_patch_to_5d(input_patch)

        with torch.no_grad():
            inputs = {'input': torch.from_numpy(input_patch).cuda()}
            output_patch = self.net(inputs)[self.output_key]
            
            # mask in gpu/cpu
            output_patch *= self.output_patch_mask

            # the model already did the crop, so no need to redo it!
            # output = self._crop_output_patch(output)
            
            output_patch = output_patch.cpu().numpy()
            return output_patch
