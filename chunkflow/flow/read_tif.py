from os import path
import tifffile

from chunkflow.chunk import Chunk
from .base import OperatorBase


class ReadTIFOperator(OperatorBase):
    def __init__(self, verbose: bool = True, name: str = 'read-tif'):
        super().__init__(name=name, verbose=verbose)

    def __call__(self, file_name, global_offset=None):
        assert path.exists(file_name)

        if self.verbose:
            print('read from file: {}'.format(file_name))

        _, file_extension = path.splitext(file_name)
        arr = tifffile.imread(file_name)

        return Chunk(arr, global_offset=global_offset)
