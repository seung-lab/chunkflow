#!/usr/bin/env python

import pznet
from .patch_inference_engine import PatchInferenceEngine


class PZNetEngine(PatchInferenceEngine):
    def __init__(self, model_file_name, net_file_name,
                 use_bn=True,
                 is_static_batch_norm=False):
        super().__init__()

        self.net = pznet.znet(model_file_name, net_file_name)

    def __call__(self, input_patch):
        """
        args:
            input_patch (5d numpy array): input patch with dimensions \
                batch/channel/z/y/x
        return:
            5d numpy array with the same dimension arrangment.
        """
        return self.net.forward(input_patch)
