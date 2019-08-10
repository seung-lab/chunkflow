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
                 patch_size: tuple = (20, 256, 256),
                 output_key: str = 'affinity',
                 num_output_channels: int = 3,
                 original_num_output_channels: int = 3,
                 patch_overlap: tuple = (4, 64, 64),
                 framework: str = 'identity',
                 batch_size: int = 1,
                 bump: str = 'wu',
                 mask_output_chunk: bool = False,
                 verbose: bool = True,
                 name: str = 'inference'):

        super().__init__(name=name, verbose=verbose)

        self.convnet_model = convnet_model
        self.convnet_weight_path = convnet_weight_path
        self.patch_size = patch_size
        self.output_key = output_key
        self.num_output_channels = num_output_channels
        self.original_num_output_channels = original_num_output_channels
        self.patch_overlap = patch_overlap
        self.framework = framework
        self.batch_size = batch_size
        self.bump = bump
        self.mask_output_chunk = mask_output_chunk

        self.engine = Engine(
            patch_size,
            patch_overlap,
            output_key=output_key,
            num_output_channels=num_output_channels,
            batch_size=batch_size,
            mask_output_chunk=mask_output_chunk,
            framework='identity',
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
