import h5py
import os

from chunkflow.chunk import Chunk
from .base import OperatorBase


class WriteH5Operator(OperatorBase):
    def __init__(self, verbose: bool = True, name: str = 'write-h5'):
        super().__init__(name=name, verbose=verbose)

    def __call__(self, chunk, file_name):
        assert '.h5' in file_name

        if self.verbose:
            print('write chunk to file: {}'.format(file_name))
        if os.path.exists(file_name):
            os.remove(file_name)

        with h5py.File(file_name) as f:
            f.create_dataset('/main', data=chunk, compression='gzip')
            if isinstance(chunk, Chunk):
                f.create_dataset('/global_offset', data=chunk.global_offset)
