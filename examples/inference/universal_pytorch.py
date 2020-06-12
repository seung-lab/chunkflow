#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from pytorch_model.model import load_model
from types import SimpleNamespace


class PatchInferencer:
    def __init__(self, model_weight_file, output_patch_mask):
        self.output_patch_mask = torch.tensor(output_patch_mask)

        patch_size = (20, 256, 256)
        patch_overlap = (4, 64, 64)
        output_key = 'affinity'
        num_output_channels = 3

        d = dict()
        d['model'] = 'rsunet'
        d['width'] = [16, 32, 64, 128]
        d['in_spec'] = {'input': (1, *patch_size)}
        d['out_spec'] = {output_key: (16, *patch_size)}
        d['scan_spec'] = {output_key: (num_output_channels, *patch_size)}
        d['pretrain'] = True 
        d['precomputed'] = False
        d['edges'] = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        d['overlap'] = tuple(patch_overlap)
        d['bump'] = None
        d['cropsz'] = None
        d['gpu_ids'] = None

        self.opt = SimpleNamespace(**d)
        self.net = load_model(self.opt, model_weight_file)
        if torch.cuda.is_available():
            self.net.cuda()
            self.output_patch_mask = self.output_patch_mask.cuda()
        assert len(self.opt.in_spec) == 1

    @property
    def compute_device(self):
        return torch.cuda.get_device_name()

    def __call__(self, input_patch):
        with torch.no_grad():
            input_patch = torch.from_numpy(input_patch).cuda()
            output_patch = self.net( input_patch ).sigmoid()
            output_patch = output_patch * self.output_patch_mask
            output_patch = output_patch.cpu().numpy()
        return output_patch
