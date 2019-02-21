

def prepare_inference_engine(convnet_model, convnet_weight_path, 
                             patch_size=(20, 256, 256), output_key='affinity',
                             num_output_channels=3,
                             original_num_output_channels=3,
                             patch_overlap=(4, 64, 64),
                             is_masked_in_device=False,
                             framework='identity', log={}):
    def _log_gpu_device():
        import torch 
        log['compute_device'] = torch.cuda.get_device_name(0)
    
    def _log_cpu_device():
        import platform
        log['compute_device'] = platform.processor() 

    # prepare for inference
    from .inference.block_inference_engine \
        import BlockInferenceEngine
    if framework == 'pznet':
        _log_cpu_device()
        from .inference.frameworks.pznet_patch_inference_engine \
            import PZNetPatchInferenceEngine
        patch_engine = PZNetPatchInferenceEngine(convnet_model, convnet_weight_path)
    elif framework == 'pytorch':
        _log_gpu_device()
        from .inference.frameworks.pytorch_patch_inference_engine \
            import PytorchPatchInferenceEngine
        patch_engine = PytorchPatchInferenceEngine(
            convnet_model,
            convnet_weight_path,
            patch_size=patch_size,
            output_key=output_key,
            num_output_channels=num_output_channels)
    elif framework == 'pytorch-multitask':
        _log_gpu_device()
        from .inference.frameworks.pytorch_multitask_patch_inference \
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
        _log_cpu_device()
        from .inference.frameworks.identity_patch_inference_engine \
            import IdentityPatchInferenceEngine
        patch_engine = IdentityPatchInferenceEngine(num_output_channels=3)
    else:
        raise Exception('invalid inference backend: {}'.format(framework))

    block_inference_engine = BlockInferenceEngine(
        patch_inference_engine=patch_engine,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        output_key=output_key,
        num_output_channels=num_output_channels,
        is_masked_in_device=is_masked_in_device)
    return block_inference_engine


def inference(chunk, block_inference_engine):
    return block_inference_engine(chunk)
