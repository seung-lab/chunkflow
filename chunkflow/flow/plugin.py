# -*- coding: utf-8 -*-
import os.path as path

import numpy as np

from .base import OperatorBase

from cloudvolume.lib import Bbox

from chunkflow.lib import load_source
from chunkflow.chunk import Chunk


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
                    '../plugins/chunkflow-plugins/chunkflowplugins', 
                    path.basename(plugin_file))

        assert path.exists(plugin_file)

        program = load_source(plugin_file)
        
        # assuming this is a func / static functor for now, maybe make it a class?
        self.execute = program.execute

    def __call__(self, inputs: list, args: str = None):
        voxel_offset = None
        voxel_size = None
        shape = None
        for inp in inputs:
            if isinstance(inp, Chunk):
                voxel_offset = inp.voxel_offset
                voxel_size = inp.voxel_size
                shape = inp.shape
                break

        if len(inputs) == 0 and args is None:
            outputs = self.execute()
        elif len(inputs) == 0 and args is not None:
            outputs = self.execute(args=args)
        elif len(inputs) > 0 and args is None:
            outputs = self.execute(*inputs)
        else:
            outputs = self.execute(*inputs, args=args) 

        # automatically convert the ndarrays to Chunks
        if voxel_offset is not None and outputs is not None:
            assert shape is not None
            for idx, output in enumerate(outputs):
                if isinstance(output, np.ndarray):
                    # in case the plugin did some symmetric cropping
                    offset = tuple(vo + (ins - outs)//2 for vo, ins, outs in zip(voxel_offset, shape[-3:], output.shape[-3:]) )
                    outputs[idx] = Chunk(output, voxel_offset=offset, voxel_size=voxel_size)
        
        return outputs