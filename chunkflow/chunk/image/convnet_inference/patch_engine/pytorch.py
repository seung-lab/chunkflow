# from .inference_engine import InferenceEngine
# import imp
from typing import Union
import torch
import numpy as np
from .base import PatchEngine
from chunkflow.lib import load_source


class PyTorch(PatchEngine):
    def __init__(self,
                 patch_size: Union[tuple, list],
                 patch_overlap: Union[tuple, list],
                 model_file_name: str,
                 weight_file_name: str,
                 use_batch_norm: bool = True,
                 is_static_batch_norm: bool = False,
                 num_output_channels: int = 3,
                 mask: np.ndarray = None):
        super().__init__(patch_size, patch_overlap)
        self.num_output_channels = num_output_channels
        if torch.cuda.is_available():
            self.is_gpu = True
            # put mask to gpu
            self.mask = torch.from_numpy(self.mask).cuda()
        else:
            self.is_gpu = False

        net_source = load_source(model_file_name)
        self.net = net_source.InstantiatedModel
        #self.net.load(weight_file_name)

        chkpt = torch.load(weight_file_name)
        state_dict = chkpt['state_dict'] if 'state_dict' in chkpt else chkpt
        self.net.load_state_dict(state_dict)

        if self.is_gpu:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(
                self.net, device_ids=range(torch.cuda.device_count()))

        # Print model's state_dict
        #print("Model's state_dict:")
        #for param_tensor in self.net.state_dict():
        #    print(param_tensor, "\t", self.net.state_dict()[param_tensor].size())

        if use_batch_norm and is_static_batch_norm:
            self.net.eval()

        if hasattr(net_source, "pre_process"):
            self.pre_process = net_source.pre_process
        else:
            self.pre_process = self._identity

        if hasattr(net_source, "post_process"):
            self.post_process = net_source.post_process
        else:
            self.post_process = self._identity
        
    def _identity(self, patch):
        return patch

    def __call__(self, input_patch):
        # make sure that the patch is 5d ndarray
        input_patch = self._reshape_patch(input_patch)

        with torch.no_grad():
            input_patch = self.pre_process(input_patch)
            input_patch = torch.from_numpy(input_patch)
            if self.is_gpu:
                input_patch = input_patch.cuda()
            
            # the network input and output should be dict
            output_patch = self.net(input_patch)

            # only transfer required channels to cpu
            # use narrow function to avoid copy.
            output_patch = self.post_process(output_patch)
            #output_patch = output_patch.narrow(1, 0, self.num_output_channels)
            #output_patch = torch.sigmoid(output_patch)
            
            #import h5py
            #with h5py.File('/tmp/patch.h5', "w") as f:
            #    f['main'] = output_patch[0,:,:,:,:].data.cpu().numpy()
            
            # mask in gpu/cpu
            output_patch *= self.mask
             
            if self.is_gpu:
                # transfer to cpu
                output_patch = output_patch.data.cpu()
            output_patch = output_patch.numpy()
            return output_patch
