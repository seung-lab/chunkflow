import tifffile

from .base import OperatorBase


class WriteTIFOperator(OperatorBase):
    def __init__(self, verbose: bool = True, name: str = 'write-tif'):
        super().__init__(name=name, verbose=verbose)

    def __call__(self, chunk, file_name, global_offset=None):
        if self.verbose:
            print('write chunk to file: ', file_name)
        tifffile.imwrite(file_name, data=chunk)
