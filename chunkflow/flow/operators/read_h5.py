from os import path
import numpy as np
import h5py

from chunkflow.chunk import Chunk
from .base import OperatorBase


class ReadH5Operator(OperatorBase):
    """Read HDF5 files"""
    def __init__(self, verbose: bool = True, name: str = 'read-h5'):
        super().__init__(name=name, verbose=verbose)

    def __call__(self, file_name: str, dataset_path: str = '/main',
                 global_offset: tuple = None):
        assert path.exists(file_name)
        assert h5py.is_hdf5(file_name)

        if self.verbose:
            print('read from HDF5 file: {}'.format(file_name))
       
        global_offset_path = path.join(path.dirname(file_name), 'global_offset')
        with h5py.File(file_name) as f:
            arr = np.asarray(f[dataset_path])
            
            if global_offset is None:
                if global_offset_path in f:
                    global_offset = tuple(f[global_offset_path])

        if self.verbose:
            print('global offset: {}'.format(global_offset))
 
        return Chunk(arr, global_offset=global_offset)
