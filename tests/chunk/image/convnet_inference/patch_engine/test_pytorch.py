from chunkflow.operators.block_inference.frameworks.pytorch_patch_inference_engine import PyTorchPatchInferenceEngine
from .rsunet import RSUNet
import os

def test_pytorch_patch_inference_engine():
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
