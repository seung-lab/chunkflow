#!/usr/bin/env python
import argparse
import numpy as np
import re
import os


re_local = re.compile('^file://')


class InferenceOptions(object):
    """
    Inference options.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._initialize()

    def parse(self):
        opt = self.parser.parse_args()

        # Model spec.
        opt.patch_size = tuple(opt.patch_size)
        opt.in_spec = dict(input=(1,) + opt.patch_size)
        opt.out_spec = {opt.output_key:
                        (opt.output_channels,) + opt.patch_size}

        # Scan spec.
        opt.scan_spec = dict(affinity=(3,) + opt.patch_size)
        assert len(opt.overlap) == 3
        assert np.all(opt.overlap >= 0.0) and np.all(opt.overlap <= 1.0)
        stride = 1.0 - opt.overlap
        opt.scan_params = dict(stride=tuple(stride), blend='bump')

        args = vars(opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        self.opt = opt
        self._prepare_dirs()

        return self.opt

    def _prepare_dirs(self):
        def make_dir(path):
            if re_local.match(path) and (not os.path.isdir(path)):
                os.makedirs(path)
        make_dir(self.input_dir)
        make_dir(self.output_dir)
        make_dir(self.exchange_dir)

    def _initialize(self):
        # input and output
        self.parser.add_argument('--input_dir', type=str, required=True,
                                 help="input directory path, \
                                 support file://, gs:// and s3:// protocols")
        self.parser.add_argument('--output_dir', type=str, required=True,
                                 help="output directory path, \
                                 support file://, gs://, s3:// protocols.")
        self.parser.add_argument('--exchange_dir', type=str, required=True,
                                 help="chunk exchange place, \
                                 support file://, gs://, s3:// protocols.")

        # task scheduler
        self.parser.add_argument('--roles_mask', type=int,
                                 help="output block face role, \
                                 [zs, ze, ys, ye, xs, xe], \
                                 1 means donor, -1 means receiver.",
                                 default=[1, 1, 1, 1, 1, 1], nargs='+')

        # Model spec.
        self.parser.add_argument('--model_path', type=str, required=True,
                                 help="the path of convnet model")
        self.parser.add_argument('--net_path', type=str, required=True,
                                 help="the path of convnet weights")
        self.parser.add_argument('--patch_size', type=int,
                                 default=[18, 256, 256], nargs='+',
                                 help="convnet input patch size")
        self.parser.add_argument('--output_channels', type=int, default=3,
                                 help="number of convnet output channels")
        self.parser.add_argument('--no_eval', action='store_true',
                                 help="this is on then using dynamic \
                                 batchnorm, otherwise static.")
        self.parser.add_argument('--output_key', type=str, default='affinity',
                                 help="the name of the final output layer")

        # resources
        self.parser.add_argument('--gpu_ids', type=str, default=['0'],
                                 nargs='*')

        # Scan spec.
        self.parser.add_argument('--overlap', type=float,
                                 default=[0.5, 0.5, 0.5], nargs='+',
                                 help="overlap percentile, \
                                 such as [0.5, 0.5, 0.5]")


if __name__ == '__main__':
    opt = InferenceOptions().parse()
