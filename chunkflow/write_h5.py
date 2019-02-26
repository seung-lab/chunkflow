import h5py

from .lib.offset_array import OffsetArray


def write_h5(chunk, file_name, verbose=True):
    assert '.h5' in file_name
    
    if verbose:
        print('write chunk to file: {}'.format(file_name))
    
    with h5py.File(file_name) as f:
        f.create_dataset('/main', data=chunk, compression='gzip')
        if isinstance(chunk, OffsetArray):
            f.create_dataset('/global_offset', 
                             data=chunk.global_offset)
