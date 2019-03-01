from .operator_base import OperatorBase
import numpy as np

from cloudvolume import view, hyperview
import neuroglancer


class ViewOperator(OperatorBase):
    def __init__(self, name='view'):
        super().__init__(name=name)

    def __call__(self, chunk, seg=None):
        """view chunk using cloudvolume view"""
        # cloudvolume use fortran order
        chunk = np.transpose(chunk)
        if seg:
            seg = np.transpose(seg)
            hyperview(chunk, seg)
        elif np.issubdtype(chunk.dtype, np.floating) or chunk.dtype == np.uint8:
            # this is an image 
            view(chunk)
        else:
            view(chunk, segmentation=True)
