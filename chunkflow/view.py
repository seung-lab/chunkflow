import numpy as np


from cloudvolume import view, hyperview


def view_chunk(chunk, seg=None):
    """view chunk using cloudvolume view"""
    if seg:
        hyperview(chunk, seg)
    elif np.issubdtype(chunk.dtype, np.floating) or chunk.dtype == np.uint8:
        # this is an image 
        view(chunk)
    else:
        view(chunk, segmentation=True)
