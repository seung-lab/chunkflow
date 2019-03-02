from .operator_base import OperatorBase
import numpy as np
from cloudvolume import CloudVolume 


class SaveOperator(OperatorBase):
    def __init__(self, volume_path, mip, verbose=True, name='save'):
        super().__init__(name=name)
        self.verbose = verbose
        self.volume = CloudVolume(
            volume_path,
            fill_missing=True,
            bounded=False,
            autocrop=True,
            mip=mip,
            progress=verbose)
 
    def __call__(self, chunk):
        chunk_slices = chunk.slices 
        # transpose czyx to xyzc order
        arr = np.transpose(chunk)
        self.volume[chunk_slices[::-1]] = arr
