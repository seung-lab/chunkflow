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
                 args: tuple = None,
                 name: str = 'plugin-1'):
        r"""
        Loads a custom python file specified in `opprogram`, which 
        should contain a callable named "exec" such that 
        a call of `op_call(chunk, args)` operates on the chunk.
        """
        super().__init__(name=name)

        self.args = args

        if not plugin_file.endswith('.py'):
            plugin_file += '.py'

        if not path.isfile(plugin_file):
            plugin_file = path.join(path.dirname(path.realpath(__file__)), '../plugins', plugin_file)

        assert path.isfile(plugin_file)

        program = load_source(plugin_file)
        
        # assuming this is a func / static functor for now, maybe make it a class?
        self.exec = program.exec  

    def __call__(self, chunk: Chunk):
        logging.info(f'{self.name} on {chunk.dtype} with shape {chunk.shape}')
        voxel_offset = chunk.voxel_offset 
        chunk = self.exec(chunk, *self.args)
        if isinstance(chunk, np.ndarray):
            if len(voxel_offset) < chunk.ndim:
                voxel_offset = tuple(0, *voxel_offset)
            elif len(voxel_offset) > chunk.ndim:
                voxel_offset = voxel_offset[1:]
            chunk = Chunk(chunk, voxel_offset=voxel_offset) 

        return chunk
