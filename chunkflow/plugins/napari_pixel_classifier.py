import os

import numpy as np
from chunkflow.chunk import Chunk

import apoc


def execute(img: Chunk, opencl_filename: str='./PixelClassifier.cl', 
        obj_id: int=None, reverse: bool = False):
    """
    Note that we need to set the following command to use the opencl backends in cluster nodes.
        module load modules/2.1.1-20230405  gcc/11.3.0 boost/1.80.0
        export OCL_ICD_VENDORS="$CONDA_PREFIX/etc/OpenCL/vendors"
        export XDG_CACHE_HOME="/tmp/opencl"
    """
    assert os.path.exists(opencl_filename), \
        f'traied classifier file not found: {opencl_filename}'
    clf = apoc.PixelClassifier(opencl_filename=opencl_filename)
    breakpoint()
    pred = clf.predict(img.array)
    pred = np.asarray(pred)

    if obj_id is not None:
        pred = (pred == obj_id)
        if reverse:
            pred = np.logical_not(pred)
    else:
        pred = pred.astype(np.uint32)

    pred = Chunk(pred)
    pred.set_properties(img.properties)
    pred.layer_type = 'segmentation'

    return pred