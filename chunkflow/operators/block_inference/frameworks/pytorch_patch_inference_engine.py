# from .inference_engine import InferenceEngine
# import imp

import torch
import numpy as np
import importlib, types
from .patch_inference_engine import PatchInferenceEngine


def load_source(fname, module_name="something"):
    """ Imports a module from source """
    loader = importlib.machinery.SourceFileLoader(module_name, fname)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod


class PytorchPatchInferenceEngine(PatchInferenceEngine):
    def __init__(self,
                 model_file_name,
                 weight_file_name,
                 use_batch_norm=True,
                 is_static_batch_norm=False,
                 output_key='affinity',
                 num_output_channels=3):
        super().__init__()
        self.num_output_channels=num_output_channels
        if torch.cuda.is_available():
            self.is_gpu = True
        else:
            self.is_gpu = False

        self.net = load_source(model_file_name).InstantiatedModel
        if self.is_gpu:
            self.net.load_state_dict(torch.load(weight_file_name))
            self.net.cuda()
            self.net = torch.nn.DataParallel(self.net,
                                             device_ids=range(torch.cuda.device_count()))
        else:
            self.net.load_state_dict(torch.load(weight_file_name, map_location='cpu'))
            
        if use_batch_norm and is_static_batch_norm:
            self.net.eval()

    def __call__(self, patch):
        # make sure that the patch is 5d ndarray
        patch = self._reshape_patch(patch)

        with torch.no_grad():
            in_v = torch.from_numpy(patch)
            if self.is_gpu:
                in_v = in_v.cuda()
            # this net returns a list, but has one output
            output_v = self.net(in_v)[0]
            # the network output do not have sigmoid function
            output_patch = torch.sigmoid(output_v).data.cpu().numpy()
            return output_patch[:,0:self.num_output_channels, :,:]


if __name__ == "__main__":
    from .rsunet import RSUNet
    import os
    model_file_name = os.path.expanduser('~/workspace/pytorch-model/pytorch_model/rsunet.py')
    net_file_name = '/nets/weight.chkpt'
    engine = PyTorchEngine(model_file_name, net_file_name)

    import h5py
    from dataprovider.emio import imsave
    fimg = '/tmp/img.h5'
    with h5py.File(fimg) as f:

        patch = f['main'][:20, :256, :256]
        patch = np.asarray(patch, dtype='float32') / 255.0
        output = engine(patch)
        print('shape of output: {}'.format(output.shape))
        imsave(output, '/tmp/patch.h5')
