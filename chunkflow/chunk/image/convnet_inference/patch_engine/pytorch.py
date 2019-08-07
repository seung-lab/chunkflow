# from .inference_engine import InferenceEngine
# import imp
from typing import Union
import torch
import numpy as np
from .base import PatchEngine
from chunkflow.lib import load_source


class PyTorch(PatchEngine):
    def __init__(self,
                 model_file_name: str,
                 weight_file_name: str,
                 patch_size: Union[tuple, list],
                 patch_overlap: Union[tuple, list],
                 use_batch_norm: bool = True,
                 is_static_batch_norm: bool = False,
                 output_key: str = 'affinity',
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

        self.net_source = load_source(model_file_name)
        self.net = self.net_source.InstantiatedModel

        if self.is_gpu:
            self.net.load_state_dict(torch.load(weight_file_name))
            self.net.cuda()
            self.net = torch.nn.DataParallel(
                self.net, device_ids=range(torch.cuda.device_count()))
        else:
            self.net.load_state_dict(
                torch.load(weight_file_name, map_location='cpu'))

        if use_batch_norm and is_static_batch_norm:
            self.net.eval()

    def __call__(self, patch):
        # make sure that the patch is 5d ndarray
        patch = self._reshape_patch(patch)

        with torch.no_grad():
            if hasattr(self.net_source, 'pre_process'):
                patch = self.net_source.pre_process(patch)
            in_v = torch.from_numpy(patch)
            if self.is_gpu:
                in_v = in_v.cuda()

            output_v = self.net(in_v)
            if hasattr(self.net_source, 'post_process'):
                output_patch = self.net_source.post_process(output_v)
            else:  # backward compatibility
                # this net returns a list, but has one output
                output_v = output_v[0]
                # the network output do not have sigmoid function
                output_patch = torch.sigmoid(output_v)

            # only transfer required channels to cpu
            output_patch = output_patch[:, 0:self.num_output_channels, :, :]
            # mask in gpu/cpu
            output_patch *= self.mask
            if self.is_gpu:
                # transfer to cpu
                output_patch = output_patch.data.cpu().numpy()
            return output_patch
