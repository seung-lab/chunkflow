import os

import numpy as np
import h5py

from chunkflow.lib.synapses import Synapses
from chunkflow.chunk import Chunk

file_path = '/mnt/ceph/users/neuro/wasp_em/jwu/sample1/72_tbar/02_tbar_points'

def execute(synapses: Synapses, seg: Chunk):
    tbars = synapses.tbars
    if len(tbars) == 0:
        print('tbars with 0 element: ', seg.bbox.to_filename())
        return

    if seg.bbox.to_filename() == '0-1024_3072-4096_4096-5120':
        return

    # if seg.bbox.to_filename() == '0-1024_5120-6144_3072-4096':
    #     breakpoint()

    tbars = tbars.astype(np.int64)
    voxel_offset = np.asarray(seg.voxel_offset, np.int64)
    tbars -= voxel_offset
    
    assert np.all(tbars >= 0)

    # eliminate the tbars 
    mask = ( tbars[:, 0]!=1024 )
    mask = np.logical_and(mask, tbars[:, 1] != 1024) 
    mask = np.logical_and(mask, tbars[:, 2] != 1024) 
    tbars = tbars[mask, :]

    objects = seg.array[tbars[:, 0], tbars[:, 1], tbars[:, 2]]
    assert len(objects) == tbars.shape[0]

    fname = os.path.join(file_path, f'{seg.bbox.to_filename()}.h5')
    with h5py.File(fname, mode='r+') as hf:
        hf['object_ids'] = objects
        hf['tbars_3'] = tbars + voxel_offset