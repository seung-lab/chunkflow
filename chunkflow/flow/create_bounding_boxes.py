from tqdm import tqdm
from itertools import product

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox

def create_bounding_boxes(chunk_size:tuple, chunk_overlap: tuple=(0,0,0),
                          roi_start: tuple=None, roi_stop: tuple=None, layer_path: str=None,
                          mip:int=0, grid_size: tuple=None, verbose: bool=True):
    if layer_path:
        vol = CloudVolume(layer_path, mip=mip)
        # dataset shape as z,y,x
        dataset_size = vol.mip_shape(mip)[:3][::-1]
        dataset_offset = vol.mip_voxel_offset(mip)[::-1]
        if roi_stop is None:
            roi_stop = Vec(*[o+s for o, s in zip(dataset_offset, dataset_size)])
        if roi_start is None:
            # note that we normally start from -overlap to keep the chunks aligned!
            roi_start = dataset_offset - chunk_overlap

    chunk_size = Vec(*chunk_size)
    chunk_overlap = Vec(*chunk_overlap)
    stride = chunk_size - chunk_overlap
    if isinstance(grid_size, tuple):
        grid_size = Vec(*grid_size)
   
    assert roi_start is not None
    if isinstance(roi_start, tuple):
        roi_start = Vec(*roi_start)
    
    if roi_stop is None:
        roi_stop = roi_start + stride*grid_size + chunk_overlap
    elif isinstance(roi_stop, tuple):
        roi_stop = Vec(*roi_stop)
    roi_size = roi_stop - roi_start

    if grid_size is None:
        grid_size = (roi_size - chunk_overlap) // stride + 1
    
    # the stride should not be zero if there is more than one chunks
    for g, s in zip(grid_size, stride):
        if g > 1:
            assert s > 0
    
    final_output_stop = roi_start + (grid_size-1) * stride + chunk_size
    if verbose:
        print('\nroi start: ', roi_start)
        print('stride: ', stride)
        print('grid size: ', grid_size)
        print('final output stop: ', final_output_stop)

    bboxes = []
    for (z, y, x) in product(range(grid_size[0]), 
                             range(grid_size[1]),
                             range(grid_size[2])):
        chunk_start = roi_start + Vec(z, y, x) * stride
        bbox = Bbox.from_delta(chunk_start, chunk_size)
        bboxes.append( bbox )
    return bboxes
