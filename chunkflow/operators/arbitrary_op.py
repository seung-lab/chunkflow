# -*- coding: utf-8 -*-

from chunkflow.chunk import Chunk
from .operator_base import OperatorBase

from .block_inference.frameworks.pytorch_patch_inference_engine import load_source


class ArbitraryChunkOperator(OperatorBase):
    r"""
    """
    def __init__(self,
                 opprogram: str=None,
                 args: str=None,
                 name: str='arbitrary-chunk-op'):
        """
        """
        super().__init__(name=name)

        self.args = args

        self.program = load_source(opprogram)
        self.call = self.program.op_call  # assuming this is a func / static functor for now, maybe make it a class?

    def __call__(self, chunk, debug=False):
        if debug:
            print(chunk.dtype, chunk.shape)

        out = self.call(chunk, self.args)

        return Chunk(out, global_offset=chunk.global_offset)
