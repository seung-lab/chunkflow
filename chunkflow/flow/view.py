import numpy as np

from cloudvolume import view, hyperview
from .base import OperatorBase


class ViewOperator(OperatorBase):
    def __init__(self, name: str = 'view', verbose: bool = True):
        super().__init__(name=name, verbose=verbose)

    def __call__(self, chunk, seg=None):
        """view chunk using cloudvolume view"""
        # cloudvolume use fortran order
        chunk = chunk.transpose()
        if seg:
            seg = seg.transpose()
            hyperview(chunk, seg)
        elif np.issubdtype(chunk.dtype,
                           np.floating) or chunk.dtype == np.uint8:
            # this is an image
            view(chunk)
        else:
            view(chunk, segmentation=True)
