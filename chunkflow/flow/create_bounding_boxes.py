from tqdm import tqdm
from itertools import product

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox

def create_bounding_boxes(chunk_size:tuple, overlap: tuple=(0,0,0),
                    start:tuple=None, layer_path: str=None, mip:int=0, 
                    grid_size: tuple=None, verbose: bool=True):
    if layer_path:
        vol = CloudVolume(layer_path, mip=mip)
        # dataset shape as z,y,x
        dataset_shape = vol.mip_shape(mip)[:3][::-1]
        dataset_offset = vol.mip_voxel_offset(mip)[::-1]

    chunk_size = Vec(*chunk_size)
    overlap = Vec(*overlap)
    stride = chunk_size - overlap

    if start is None:
        # note that we normally start from -overlap to keep the chunks aligned!
        start = dataset_offset - overlap
        volume_size = dataset_shape
    else:
        start = Vec(*start)

    if grid_size is None:
        volume_size = dataset_shape - (start - dataset_offset)
        grid_size = (volume_size-overlap) // stride + 1

    # the stride should not be zero if there is more than one chunks
    for g, s in zip(grid_size, stride):
        if g > 1:
            assert s > 0

    if verbose:
        print('\nstart: ', start)
        print('stride: ', stride)
        print('grid size: ', grid_size)
        print('chunk_size: ', chunk_size, '\n')

    bboxes = []
    for (z, y, x) in tqdm(product(range(grid_size[0]), range(grid_size[1]),
                                                       range(grid_size[2]))):
        chunk_start = start + Vec(z, y, x) * stride
        bbox = Bbox.from_delta(chunk_start, chunk_size)
        bboxes.append( bbox )

    return bboxes