from .base import OperatorBase
from chunkflow.chunk.image.convnet_inference.engine import Engine


class InferenceOperator(OperatorBase):
    """
    bump: (str) bump function of mask. {wu, zung} 
            this only works with the pytorch-multitask backend.
    """
    def __init__(self,
                 convnet_model: str,
                 convnet_weight_path: str,
                 input_patch_size: tuple = (20, 256, 256),
                 output_patch_size: tuple = (20, 256, 256),
                 num_output_channels: int = 3,
                 output_patch_overlap: tuple = (4, 64, 64),
                 framework: str = 'identity',
                 batch_size: int = 1,
                 bump: str = 'wu',
                 mask_output_chunk: bool = False,
                 verbose: bool = False,
                 name: str = 'inference'):

        super().__init__(name=name, verbose=verbose)
        self.framework = framework

        self.engine = Engine(
            convnet_model,
            convnet_weight_path,
            input_patch_size=input_patch_size,
            output_patch_size=output_patch_size,
            output_patch_overlap=output_patch_overlap,
            num_output_channels=num_output_channels,
            framework=framework,
            batch_size=batch_size,
            bump=bump,
            mask_output_chunk=mask_output_chunk,
            verbose=verbose)

    def __call__(self, chunk):
        """ the chunk size should always be the same 
        and the size is aligned with patch size and patch overlap
        """
        return self.engine(chunk)

    @property
    def compute_device(self):
        if self.framework in ('pznet', 'identity'):
            import platform
            return platform.processor()
        else:
            import torch
            return torch.cuda.get_device_name(0)
