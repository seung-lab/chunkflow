import os
import torch
from pytorch_model.model import load_model
from types import SimpleNamespace

from .base import PatchEngine


class PytorchMultitask(PatchEngine):
    def __init__(self,
                 convnet_model,
                 convnet_weight_path,
                 patch_size=(20, 256, 256),
                 width=(16, 32, 64, 128),
                 output_key='affinity',
                 original_num_output_channels=3,
                 num_output_channels=3,
                 bump='wu',
                 patch_overlap=(4, 64, 64)):
        """
        this function do masking in gpu for speed up, 
        so we need the patch_overlap information.
        """
        super().__init__()

        # we currently only support two types of model
        assert convnet_model in ('rsunet', 'rsunet_act')

        self.output_key = output_key

        d = {
            'model': convnet_model,
            'width': width,
            'in_spec': {
                'input': (1, *patch_size)
            },
            'out_spec': {
                output_key: (original_num_output_channels, *patch_size)
            },
            'scan_spec': {
                output_key: (num_output_channels, *patch_size)
            },
            'pretrain': True,
            'precomputed': torch.cuda.is_available(),
            'edges': [(0, 0, 1), (0, 1, 0), (1, 0, 0)],
            'overlap': patch_overlap,
            'bump': bump
        }

        self.opt = SimpleNamespace(**d)
        assert os.path.isfile(convnet_weight_path)
        self.net = load_model(self.opt, convnet_weight_path)
        assert len(self.opt.in_spec) == 1

    def __call__(self, patch):
        # make sure that patch is 5d ndarray
        patch = self._reshape_patch(patch)

        with torch.no_grad():
            inputs = dict()
            for k in sorted(self.opt.in_spec):
                # assume that only one input
                inputs[k] = torch.from_numpy(patch).cuda()
            outputs = self.net(inputs)
            output = outputs[self.output_key].cpu().numpy()
            return output
