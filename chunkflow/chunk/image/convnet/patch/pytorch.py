# from .inference_engine import InferenceEngine
# import imp
import torch
from .base import PatchInferencerBase
from chunkflow.lib import load_source

torch.backends.cudnn.benchmark = True


class PyTorch(PatchInferencerBase):
    """perform inference for an image patch using pytorch.
    Parameters
    ----------
    patch_size: size of input/output patch size. We assume that 
        the input and output patch size is the same. 
    patch_overlap: overlap of neighboring patches.
    model_file_name: file name of model
    weight_file_name: file name of trained weight.
    num_output_channels: number of output channels.
    
    You can make some customized processing in your model file. 
    You can define `load_model` function to customize your way of 
    loading model. This is useful for loading some models trained using
    old version pytorch (<=0.4.0). You can also define `pre_process` 
    and `post_process` function to insert your own customized processing.
    """
    def __init__(self, convnet_model: str, convnet_weight_path: str,
                 input_patch_size: tuple, 
                 output_patch_size: tuple, 
                 output_patch_overlap: tuple,
                 num_output_channels: int = 1, 
                 dtype: str='float32',
                 bump: str='wu'):
        # To-Do: support zung function
        assert bump == 'wu'
        super().__init__(input_patch_size, output_patch_size, 
                         output_patch_overlap, num_output_channels, 
                         dtype=dtype)

        self.num_output_channels = num_output_channels
        if torch.cuda.is_available():
            self.is_gpu = True
            # put mask to gpu
            self.output_patch_mask = torch.from_numpy(self.output_patch_mask).cuda()
        else:
            self.is_gpu = False

        net_source = load_source(convnet_model)

        if hasattr(net_source, "load_model"):
            self.model = net_source.load_model(convnet_weight_path)
        else:
            self.model = net_source.InstantiatedModel
            chkpt = torch.load(convnet_weight_path)
            state_dict = chkpt['state_dict'] if 'state_dict' in chkpt else chkpt
            self.model.load_state_dict(state_dict)
        
        if self.is_gpu and next(self.model.parameters()).is_cuda:
            self.model.cuda()

            # data parallel do not work with old emvision net
            #self.model = torch.nn.DataParallel(
            #    self.model, device_ids=range(torch.cuda.device_count()))

        # Print model's state_dict
        #print("Model's state_dict:")
        #for param_tensor in self.model.state_dict():
        #    print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())


        if hasattr(net_source, "pre_process"):
            self.pre_process = net_source.pre_process
        else:
            self.pre_process = self._pre_process

        if hasattr(net_source, "post_process"):
            self.post_process = net_source.post_process
        else:
            self.post_process = self._identity
    
    @property
    def compute_device(self):
        return torch.cuda.get_device_name(0)

    def _pre_process(self, input_patch):
        input_patch = torch.from_numpy(input_patch)
        if self.is_gpu:
            input_patch = input_patch.cuda()
        return input_patch
    
    def _identity(self, patch):
        return patch

    def __call__(self, input_patch):
        # make sure that the patch is 5d ndarray
        input_patch = self._reshape_patch_to_5d(input_patch)

        with torch.no_grad():
            net_input = self.pre_process(input_patch)
            # the network input and output should be dict
            net_output = self.model(net_input)

            # get the required output patch from network 
            # The processing depends on network model and application
            output_patch = self.post_process(net_output)

            # mask in gpu/cpu
            output_patch = self._crop_output_patch(output_patch)
            output_patch *= self.output_patch_mask

            if self.is_gpu:
                # transfer to cpu
                output_patch = output_patch.data.cpu()
            output_patch = output_patch.numpy()
            return output_patch
