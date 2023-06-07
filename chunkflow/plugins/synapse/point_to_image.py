import os


import numpy as np
import h5py
from scipy import misc

from chunkflow.lib.cartesian_coordinate import BoundingBox

DIR = '/mnt/ceph/users/neuro/wasp_em/jwu/sample1/72_tbar/02_tbar_points'

def execute(bbox: BoundingBox):

    fname = os.path.join(DIR, f'{bbox.to_filename()}.h5')
    if os.path.getsize(fname) > 0:
        with h5py.File(fname, 'r') as hf:
            tbars = np.asarray(hf['tbars'])

        img = np.zeros((12000, 8000), dtype=np.uint8)
        for idx in range(tbars.shape[0]):
            
        
