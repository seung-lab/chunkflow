# -*- coding: utf-8 -*-
import os.path as path

from .base import OperatorBase

from cloudvolume.lib import Bbox

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

    def __call__(self, inputs, bbox: Bbox = None, args: str = None):
        if args is None and bbox is None:
            return self.exec(*inputs)
        if args is None and bbox is not None:
            return self.exec(*inputs, bbox=bbox)
        if args is not None and bbox is None:
            return self.exec(*inputs, args=args) 
        return self.exec(*inputs, bbox=bbox, args=args)
