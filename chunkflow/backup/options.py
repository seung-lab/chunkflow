#!/usr/bin/env python
import argparse
import numpy as np
import re
import os
import math

re_local = re.compile('^file://')


class BaseOptions(object):
    """
    Inference options.
    """

    def __init__(self):
        self._add_base_arguments()

    def parse(self):
        self.opt = self.parser.parse_args()
        self._setup_base_options()
        self._setup()
        self.print_args()
        return self.opt

    def print_args(self):
        print('------------ Options -------------')
        for k, v in sorted(vars(self.opt).items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _setup_base_options(self):
        self._make_dir(self.opt.input_dir)
        self._make_dir(self.opt.output_dir)
        self._make_dir(self.opt.exchange_dir)
        self.opt.output_block_slices = tuple(
            slice(start, start + size) for start, size in zip(
                self.opt.output_block_start, self.opt.output_block_size))

    def _setup(self):
        """
        virtual function for configuration of parameters
        """
        raise NotImplementedError()

    def _make_dir(self, path):
        if re_local.match(path) and (not os.path.isdir(path)):
            os.makedirs(path.replace('file://', ''))

    def _add_base_arguments(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # input and output
        self.parser.add_argument(
            '--output_dir',
            type=str,
            required=True,
            help="output directory path, \
                                 support file://, gs://, s3:// protocols.")
        self.parser.add_argument(
            '--exchange_dir',
            type=str,
            required=True,
            help="chunk exchange place, \
                                 support file://, gs://, s3:// protocols.")
        self.parser.add_argument(
            '--output_block_start',
            type=int,
            help="the start coordinate of output block",
            required=True,
            nargs='+')
        self.parser.add_argument(
            '--output_block_size',
            type=int,
            help="the size of output block",
            default=[112, 1152, 1152],
            nargs='+')
        self.parser.add_argument(
            '--output_channels',
            type=int,
            default=3,
            help="number of convnet output channels")
        self.parser.add_argument(
            '--overlap',
            type=int,
            default=[4, 64, 64],
            nargs='+',
            help="overlap by number of voxels")


class InferenceAndDonateOptions(BaseOptions):
    """
    Inference options.
    """

    def __init__(self):
        super().__init__()
        self._add_inference_donate_arguments()

    def _setup(self):
        # Model spec.
        self.opt.patch_size = tuple(self.opt.patch_size)

        assert len(self.opt.overlap) == 3
        for o in self.opt.overlap:
            assert o >= 1
        return self.opt

    def _add_inference_donate_arguments(self):
        # receive and blend step do not need to input image chunks
        self.parser.add_argument(
            '--input_dir',
            type=str,
            required=True,
            help="input directory path, \
                                 support file://, gs:// and s3:// protocols")

        # Model spec.
        self.parser.add_argument(
            '--model_path',
            type=str,
            required=True,
            help="the path of convnet model")
        self.parser.add_argument(
            '--net_path',
            type=str,
            required=True,
            help="the path of convnet weights")
        self.parser.add_argument(
            '--patch_size',
            type=int,
            default=[32, 256, 256],
            nargs='+',
            help="convnet input patch size")
        self.parser.add_argument(
            '--no_eval',
            action='store_true',
            help="this is on then using dynamic \
                                 batchnorm, otherwise static.")
        self.parser.add_argument(
            '--output_key',
            type=str,
            default='affinity',
            help="the name of the final output layer")

        # resources
        self.parser.add_argument(
            '--framework',
            type=str,
            default='pytorch',
            help="backend of deep learning framework, \
                                 such as pytorch and pznet.")
        self.parser.add_argument('--gpu_ids', type=int, default=[0], nargs='*')


class ReceiveAndBlendOptions(BaseOptions):
    """
    receive and blend options.
    """

    def __init__(self):
        super().__init__()

    def _add_receive_blend_arguments(self):
        pass


if __name__ == '__main__':
    print('show inference and donate script options:')
    opt = InferenceAndDonateOptions().parse()
