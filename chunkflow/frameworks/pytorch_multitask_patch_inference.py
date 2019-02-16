import os
import torch
import numpy as np
from pytorch_model.model import load_model
from types import SimpleNamespace

from .patch_inference_engine import PatchInferenceEngine


class PytorchMultitaskPatchInferenceEngine(PatchInferenceEngine):
    def __init__(self,
                 convnet_model,
                 convnet_weight_path,
                 use_bn=True,
                 is_static_batch_norm=False,
                 patch_size=(20, 256, 256),
                 width=(16, 32, 64, 128),
                 output_key='affinity',
                 num_output_channels=3,
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
            'in_spec': {'input': (1, *patch_size)},
            'out_spec': {output_key: (num_output_channels, *patch_size)},
            'scan_spec': {output_key: (num_output_channels, *patch_size)},
            'pretrain': True,
            'precomputed': True,
            'edges': [(0, 0, 1), (0, 1, 0), (1, 0, 0)],
            'overlap': patch_overlap,
            'bump': 'wu'
        }

        self.opt = SimpleNamespace(**d)
        assert os.path.isfile(convnet_weight_path)
        self.net = load_model(self.opt, convnet_weight_path)
        assert len(self.opt.in_spec) == 1

    def __call__(self, patch):
        # patch should be a 5d np array
        #assert isinstance(patch, np.ndarray)
        if patch.ndim == 3:
            patch = patch.reshape((1, 1) + patch.shape)
        elif patch.ndim == 4:
            patch = patch.reshape((1, ) + patch.shape)

        with torch.no_grad():
            inputs = dict()
            for k in sorted(self.opt.in_spec):
                # assume that only one input
                inputs[k] = torch.from_numpy(patch).cuda()
            outputs = self.net(inputs)
            output = outputs[self.output_key].cpu().numpy()
            #output = np.squeeze(output)
            return output
