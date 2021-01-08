# -*- coding: utf-8 -*-
import logging
import os.path as path

import numpy as np
from chunkflow.chunk import Chunk
from .base import OperatorBase

from chunkflow.lib import load_source


class Plugin(OperatorBase):
    r"""
    Chunk operation using a custom python file.
    """
    def __init__(self, plugin_file: str,
                 name: str = 'plugin-1'):
        r"""
        Loads a custom python file specified in `opprogram`, which 
        should contain a callable named "exec" such that 
        a call of `op_call(chunk, args)` operates on the chunk.
        """
        super().__init__(name=name)


        if not plugin_file.endswith('.py'):
            plugin_file += '.py'

        if not path.exists(plugin_file):
            plugin_file = path.join(
                path.dirname(path.realpath(__file__)), 
                '../plugins', 
                path.basename(plugin_file))
            if not path.exists(plugin_file):
                plugin_file = path.join(
                    path.dirname(path.realpath(__file__)), 
                    '../plugins/chunkflow-plugins', 
                    path.basename(plugin_file))

        assert path.exists(plugin_file)

        program = load_source(plugin_file)
        
        # assuming this is a func / static functor for now, maybe make it a class?
        self.exec = program.exec  

    def __call__(self, inputs):
        voxel_offset = None
        voxel_size = None
        for input in inputs:
            if isinstance(input, Chunk):
                voxel_size = input.voxel_size
                voxel_offset = input.voxel_offset
                break

        outputs = self.exec(*inputs)
        ret = []
        for output in outputs:
            if isinstance(output, np.ndarray) and voxel_offset is not None:
                output = Chunk(output, voxel_offset=voxel_offset, voxel_size=voxel_size)
            ret.append(output)
        return ret
