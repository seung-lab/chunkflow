# from .inference_engine import InferenceEngine
# import imp
import torch
import numpy as np
from .rsunet import RSUNet
from .patch_inference_engine import PatchInferenceEngine


class PyTorchEngine(PatchInferenceEngine):
#class PyTorchEngine(object):
    def __init__(self, model_file_name, net_file_name,
                 use_bn = True,
                 is_static_batch_norm=False):
        super(PyTorchEngine, self).__init__()

        # self.net = torch.nn.DataParallel(imp.load_source(
        #            "Model", model_file_name).InstantiatedModel).cuda()
        # self.net = InferenceNet({'input':(1,18,256,256)},
        #                        {'affinity':(16,18,256,256)},5, use_bn=True).cuda()
        self.net = RSUNet({'input':(1, 18, 256, 256)},
                          {'affinity':(16, 18, 256, 256)}, 5, use_bn=use_bn).cuda()
        # self.net = imp.load_source("RSUNet",model_file_name).RSUNet(
        #    {'input':(1,18,256,256)},{'affinity':(16,18,256,256)},5).cuda()
        self.net.load_state_dict(torch.load(net_file_name))
        if use_bn and is_static_batch_norm:
            self.net.eval()
        # self.net.load(net_file_name)

    def __call__(self, patch):
        # patch should be a 5d np array
        if patch.ndim == 3:
            patch = patch.reshape((1, 1)+patch.shape)
        elif patch.ndim == 4:
            patch = patch.reshape((1, )+patch.shape)
        in_v = torch.autograd.Variable(torch.from_numpy(patch),
                                       volatile=True).cuda()
        output_v = self.net(in_v)[0] #this net returns a list, but has one output
        output_patch = torch.nn.functional.sigmoid(output_v).data.cpu().numpy()
        # remove the batch number dimension
        output_patch = np.squeeze(output_patch, axis=0)
        return output_patch

if __name__ == "__main__":
    # model_file_name = '/import/w4_wt_focused_760k_pytorch/w4_wt_focused_760k.py'
    # net_file_name = '/import/w4_wt_focused_760k_pytorch/w4_wt_focused_760k.chkpt'
    #model_file_name = '/usr/people/jingpeng/seungmount/research/kisuklee/Workbench/torms3/pinky-pytorch/code/rsunet.py'
    model_file_name = './rsunet.py'
    net_file_name = './model200000.chkpt'
    engine = PyTorchEngine(model_file_name, net_file_name)

    import h5py
    from dataprovider.emio import imsave
    fimg = '/usr/people/jingpeng/seungmount/research/kisuklee/Workbench/deep_learning/kaffe/datasets/pinky/ground_truth/stitched/img.h5'
    with h5py.File(fimg) as f:
        patch = f['main'][:18,:256,:256]
        patch = np.asarray(patch, dtype='float32') / 255.0
        output = engine(patch)
        print('shape of output: {}'.format(output.shape))
        imsave(output, '/tmp/patch.h5')


