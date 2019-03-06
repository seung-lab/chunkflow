from .operator_base import OperatorBase


class InferenceOperator(OperatorBase):
    def __init__(self, convnet_model: str, convnet_weight_path: str, 
                 patch_size=(20, 256, 256), output_key: str='affinity',
                 num_output_channels: int=3,
                 original_num_output_channels: int=3,
                 patch_overlap=(4, 64, 64),
                 framework: str='identity',
                 batch_size=1,
                 verbose: bool=True, name: str='inference'):

        super().__init__(name=name)
        
        self.framework = framework

        # prepare for inference
        from .block_inference.block_inference_engine \
            import BlockInferenceEngine
        if framework == 'pznet':
            is_masked_in_device=False
            from .block_inference.frameworks.pznet_patch_inference_engine \
                import PZNetPatchInferenceEngine
            patch_engine = PZNetPatchInferenceEngine(convnet_model, convnet_weight_path)
        elif framework == 'pytorch':
            is_masked_in_device=False
            from .block_inference.frameworks.pytorch_patch_inference_engine \
                import PytorchPatchInferenceEngine
            patch_engine = PytorchPatchInferenceEngine(
                convnet_model,
                convnet_weight_path,
                patch_size=patch_size,
                output_key=output_key,
                num_output_channels=num_output_channels)
        elif framework == 'pytorch-multitask':
            is_masked_in_device=True
            from .block_inference.frameworks.pytorch_multitask_patch_inference \
                import PytorchMultitaskPatchInferenceEngine
            patch_engine = PytorchMultitaskPatchInferenceEngine(
                convnet_model,
                convnet_weight_path,
                patch_size=patch_size,
                output_key=output_key,
                patch_overlap=patch_overlap,
                original_num_output_channels=original_num_output_channels,
                num_output_channels=num_output_channels)
        elif framework == 'identity':
            is_masked_in_device=False
            from .block_inference.frameworks.identity_patch_inference_engine \
                import IdentityPatchInferenceEngine
            patch_engine = IdentityPatchInferenceEngine(num_output_channels=3)
        else:
            raise Exception('invalid inference backend: {}'.format(framework))
        
        assert isinstance(is_masked_in_device, bool)
        self.block_inference_engine = BlockInferenceEngine(
            patch_inference_engine=patch_engine,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            output_key=output_key,
            num_output_channels=num_output_channels,
            is_masked_in_device=is_masked_in_device,
            batch_size=batch_size,
            verbose=verbose)
 
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
