from .operator_base import OperatorBase
from .block_inference.block_inference_engine import BlockInferenceEngine


class InferenceOperator(OperatorBase):
    """
    bump: (str) bump function of mask. {wu, zung} 
            this only works with the pytorch-multitask backend.
    """
    def __init__(self, convnet_model: str, convnet_weight_path: str, 
                 patch_size: tuple=(20, 256, 256), output_key: str='affinity',
                 num_output_channels: int=3,
                 original_num_output_channels: int=3,
                 patch_overlap: tuple=(4, 64, 64),
                 framework: str='identity',
                 batch_size: int=1,
                 bump: str='wu',
                 mask_output_chunk: bool=False,
                 verbose: bool=True, name: str='inference'):

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

        # mask in cpu in default
        mask_in_device = False

        self._prepare_patch_engine()
       
        self.block_inference_engine = BlockInferenceEngine(
            patch_inference_engine=self.patch_engine,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            output_key=output_key,
            num_output_channels=num_output_channels,
            batch_size=batch_size,
            mask_in_device=mask_in_device,
            mask_output_chunk=mask_output_chunk,
            verbose=verbose)
         
    def _prepare_patch_engine(self):
        # prepare for inference
        if self.framework == 'pznet':
            from .block_inference.frameworks.pznet_patch_inference_engine \
                import PZNetPatchInferenceEngine
            self.patch_engine = PZNetPatchInferenceEngine(
                self.convnet_model, self.convnet_weight_path)
        elif self.framework == 'pytorch':
            from .block_inference.frameworks.pytorch_patch_inference_engine \
                import PytorchPatchInferenceEngine
            self.patch_engine = PytorchPatchInferenceEngine(
                self.convnet_model,
                self.convnet_weight_path,
                output_key=self.output_key,
                num_output_channels=self.num_output_channels,
                device=self.compute_device)
        elif self.framework == 'pytorch-multitask':
            # currently only this type of task support mask in device
            mask_in_device = True
            from .block_inference.frameworks.pytorch_multitask_patch_inference \
                import PytorchMultitaskPatchInferenceEngine
            self.patch_engine = PytorchMultitaskPatchInferenceEngine(
                self.convnet_model,
                self.convnet_weight_path,
                patch_size=self.patch_size,
                output_key=self.output_key,
                patch_overlap=self.patch_overlap,
                original_num_output_channels=self.original_num_output_channels,
                num_output_channels=self.num_output_channels,
                bump=self.bump)
        elif self.framework == 'identity':
            from .block_inference.frameworks.identity_patch_inference_engine \
                import IdentityPatchInferenceEngine
            self.patch_engine = IdentityPatchInferenceEngine(
                num_output_channels=self.num_output_channels)
        else:
            raise Exception('invalid inference backend: {}'.format(self.framework))
 
    def __call__(self, chunk):
        # the chunk size should always be the same 
        # and the size is aligned with patch size and patch overlap
        return self.block_inference_engine(chunk)

    @property
    def compute_device(self):
        if self.framework in ('pznet', 'identity'):
            import platform
            return platform.processor() 
        else:
            import torch 
            return torch.cuda.get_device_name(0)
