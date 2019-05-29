from os import path
import numpy as np
import h5py 
import tifffile

from chunkflow.chunk import Chunk
from .operator_base import OperatorBase


class ReadFileOperator(OperatorBase):
    def __init__(self, verbose: bool=True, name: str='read-file'):
        super().__init__(name=name, verbose=verbose)

    def __call__(self, file_name, global_offset=None):
        assert path.exists(file_name)

        if self.verbose:
            print('read from file: {}'.format(file_name))

        base_name, file_extension = path.splitext(file_name)
        if file_extension=='.h5' or file_extension=='.hdf5':
            with h5py.File(file_name) as f:
                arr = np.asarray(f['/main'])

                if global_offset is None:
                    global_offset = tuple(f['/global_offset'])
                    if self.verbose:
                        print('read out global offset: {}'.format(global_offset))

        elif '.tif' in file_extension:
            arr = tifffile.imread(file_name)
        else:
            raise NotImplementedError('only support .tif, .h5 or .hdf5 files, '+
                                      'but getting {}'.format(file_extension))

        return Chunk(arr, global_offset=global_offset)
