#!/usr/bin/env python
import sys
from .base import PatchEngine


class PZNet(PatchEngine):
    def __init__(self, net_dir, use_bn=True, is_static_batch_norm=False,
                 dtype: str='float32'):
        super().__init__(dtype=dtype)
        sys.path.append(net_dir)
        import pznet
        # self.net = pznet.znet(model_file_name, net_file_name)
        self.net = pznet.znet()
        self.net.load_net(net_dir)
    
    @property
    def compute_device(self):
        return platform.processor()
    
    def __call__(self, patch):
        """
        args:
            patch (5d numpy array): input patch with dimensions \
                batch/channel/z/y/x
        return:
            5d numpy array with the same dimension arrangment.
        """
        # make sure that the input patch is 5d ndarray
        patch = self._reshape_patch(patch)
        return self.net.forward(patch)


if __name__ == "__main__":
    import numpy as np
    for i in range(10):
        print("patch iteration: {}".format(i))
        engine = PZNet("/nets/pinky100/unet4-long/mip1/cores2")
        input_patch = np.random.randint(0,
                                        255, (1, 1, 20, 256, 256),
                                        dtype='uint8')
        engine(input_patch)
