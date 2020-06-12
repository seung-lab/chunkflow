#!/usr/bin/env python
import sys
import platform


class PatchInferencer:
    def __init__(self, model_weight_file, output_patch_mask):
        self.output_patch_mask = output_patch_mask

        sys.path.append(model_weight_file)
        import pznet

        # self.net = pznet.znet(model_file_name, net_file_name)
        self.net = pznet.znet()
        self.net.load_net(model_weight_file)
    
    @property
    def compute_device(self):
        return platform.processor()
    
    def __call__(self, input_patch):
        """
        args:
            patch (5d numpy array): input patch with dimensions \
                batch/channel/z/y/x
        return:
            5d numpy array with the same dimension arrangment.
        """
        # make sure that the input patch is 5d ndarray
        output_patch = self.net.forward(input_patch)
        output_patch *= self.output_patch_mask
        return output_patch
