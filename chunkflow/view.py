import numpy as np


from cloudvolume import view, hyperview
import neuroglancer


def view_chunk(chunk, seg=None, tool='cloudvolume'):
    if tool == 'cloudvolume':
        view_chunk_using_cloudvolume(chunk, seg=seg)
    elif tool == 'neuroglancer':
        view_chunk_using_neuroglancer(chunk, seg=seg)
    else:
        raise ValueError('only support two tools: cloudvolume and neuroglancer, ' + 
                         'but getting ', tool)


def view_chunk_using_cloudvolume(chunk, seg=None):
    """view chunk using cloudvolume view"""
    if seg:
        hyperview(chunk, seg)
    elif np.issubdtype(chunk.dtype, np.floating) or chunk.dtype == np.uint8:
        # this is an image 
        view(chunk)
    else:
        view(chunk, segmentation=True)
