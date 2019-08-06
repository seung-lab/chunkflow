# -*- coding: utf-8 -*-

from chunkflow.chunk import Chunk
from .operator_base import OperatorBase

from .block_inference.frameworks.pytorch_patch_inference_engine import load_source


class ArbitraryChunkOperator(OperatorBase):
    r"""
    Chunk operation using a custom python file.
    """
    def __init__(self,
                 opprogram: str=None,
                 args: str=None,
                 name: str='custom-operator-1',
                 verbose: bool=False):
        r"""
        Loads a custom python file specified in `opprogram`, which 
        should contain a callable named "op_call" such that 
        a call of `op_call(chunk, args)` operates on the chunk.
        """
        super().__init__(name=name, verbose=verbose)

        self.args = args

        self.program = load_source(opprogram)
        self.call = self.program.op_call  # assuming this is a func / static functor for now, maybe make it a class?

    def __call__(self, chunk, debug=False):
        if self.verbose:
            print(self.name, ' on ', chunk.dtype, chunk.shape)

        out = self.call(chunk, self.args)

        return Chunk(out, global_offset=chunk.global_offset)
