#!/usr/bin/env python
import sys
from chunkflow.frameworks.patch_inference_engine import PatchInferenceEngine


class PZNetPatchInferenceEngine(PatchInferenceEngine):
    def __init__(self, net_dir, use_bn=True, is_static_batch_norm=False):
        super().__init__()
        sys.path.append(net_dir)
        import pznet
        # self.net = pznet.znet(model_file_name, net_file_name)
        self.net = pznet.znet()
        self.net.load_net(net_dir)

    def __call__(self, input_patch):
        """
        args:
            input_patch (5d numpy array): input patch with dimensions \
                batch/channel/z/y/x
        return:
            5d numpy array with the same dimension arrangment.
        """
        return self.net.forward(input_patch)


if __name__ == "__main__":
    import numpy as np
    for i in range(10):
        print("patch iteration: {}".format(i))
        engine = PZNetPatchInferenceEngine(
            "/nets/pinky100/unet4-long/mip1/cores2")
        input_patch = np.random.randint(
            0, 255, (1, 1, 20, 256, 256), dtype='uint8')
        engine(input_patch)
