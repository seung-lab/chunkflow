from os import path
import numpy as np
import h5py 
import tifffile

from .lib.offset_array import OffsetArray


def read_file(file_name, global_offset=None, verbose=True):
    assert path.exists(file_name)

    if verbose:
        print('read from file: {}'.format(file_name))

    base_name, file_extension = path.splitext(file_name)
    if file_extension=='.h5' or file_extension=='.hdf5':
        with h5py.File(file_name) as f:
            arr = np.asarray(f['/main'])

            if global_offset is None:
                global_offset = f.get('/global_offset', default=None)
                if isinstance(global_offset, h5py._hl.dataset.Dataset):
                    global_offset = tuple(global_offset)
                if verbose:
                    print('global offset: {}'.format(global_offset))

    elif '.tif' in file_extension:
        arr = tifffile.imread(file_name)
    else:
        raise NotImplementedError('only support .tif, .h5 or .hdf5 files, '+
                                  'but getting {}'.format(file_extension))

    return OffsetArray(arr, global_offset=global_offset)
