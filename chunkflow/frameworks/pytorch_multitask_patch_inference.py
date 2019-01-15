# from .inference_engine import InferenceEngine
# import imp

import torch
import numpy as np
import importlib, types 
from pytorch_model.model import load_model 
from types import SimpleNamespace

from .patch_inference_engine import PatchInferenceEngine

class PytorchMultitaskPatchInferenceEngine(PatchInferenceEngine):
    def __init__(self, model_file_name, weight_file_name,
                 use_bn=True, is_static_batch_norm=False,
                 patch_size=(20, 256, 256), width=(16,32,64,128), 
                 output_key='affinity', num_output_channels=3,
                 patch_overlap=(4,64,64)):
        """
        this function do masking in gpu for speed up, 
        so we need the patch_overlap information.
        """
        super().__init__()
        self.output_key = output_key

        d = dict()
        d['width'] = list(width)
        d['in_spec'] = {'input': (1, *patch_size)}
        d['out_spec'] = {output_key: (3,*patch_size)}
        d['scan_spec'] = {output_key: (num_output_channels, *patch_size)}
        d['pretrain'] = True 
        d['precomputed'] = True
        d['edges'] = [(0,0,1),(0,1,0),(1,0,0)]
        d['overlap'] = tuple(patch_overlap)
        d['bump'] = 'wu'

        self.opt = SimpleNamespace(**d)
        self.net = load_model(self.opt, weight_file_name)
        assert len(self.opt.in_spec) == 1

    def __call__(self, patch):
        # patch should be a 5d np array
        #assert isinstance(patch, np.ndarray)
        if patch.ndim == 3:
            patch = patch.reshape((1, 1)+patch.shape)
        elif patch.ndim == 4:
            patch = patch.reshape((1, )+patch.shape)

        with torch.no_grad():
            inputs = dict()
            for k in sorted(self.opt.in_spec):
                # assume that only one input
                inputs[k] = torch.from_numpy(patch).cuda()
            outputs = self.net( inputs )
            output = outputs[self.output_key].cpu().numpy()
            #output = np.squeeze(output)
            return output

