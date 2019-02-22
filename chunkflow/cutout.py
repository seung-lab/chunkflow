import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox

from .lib.validate import validate_by_template_matching
from .igneous.downsample import downsample_with_averaging
from .lib.offset_array import OffsetArray


def cutout(chunk_slices, volume_path, output_bbox, 
           mip=0, 
           expand_margin_size=(0,0,0),
           show_progress=True, 
           fill_missing=False, validate_mip=None):
    
    chunk_slices = tuple(slice(s.start - m, s.stop + m) 
                         for s, m in zip(output_bbox.to_slices(), expand_margin_size))
    vol = CloudVolume(
        volume_path,
        bounded=False,
        fill_missing=fill_missing,
        progress=show_progress,
        mip=mip,
        parallel=False)
    # always reverse the indexes since cloudvolume use x,y,z indexing
    chunk = vol[chunk_slices[::-1]]
    # the cutout is fortran ordered, so need to transpose and make it C order
    chunk = np.transpose(chunk)
    chunk = np.ascontiguousarray(chunk)
    # if the channel number is 1, squeeze it as 3d array
    # this should not be neccessary 
    # TODO: remove this step and use 4D array all over this package.
    # always use 4D array will simplify some operations
    chunk = np.squeeze(chunk, axis=0)
    global_offset = tuple(s.start for s in chunk_slices)

    chunk = OffsetArray(chunk, global_offset=global_offset)
   
    if validate_mip:
        print('validate chunk in mip {}'.format(validate_mip))
        _validate_chunk(chunk, vol, volume_path, chunk_mip=mip, validate_mip=validate_mip,
                        fill_missing=fill_missing, show_progress=show_progress)
    return chunk

def _validate_chunk(chunk, vol, volume_path, chunk_mip=0, validate_mip=5,
                    fill_missing=False, show_progress=True):
    """
    check that all the input voxels was downloaded without black region  
    We have found some black regions in previous inference run, 
    so hopefully this will solve the problem.
    """
    assert validate_mip >= chunk_mip
    # only use the region corresponds to higher mip level
    # clamp the surrounding regions in XY plane
    # this assumes that the input dataset was downsampled starting from the
    # beginning offset in the info file
    global_offset = chunk.global_offset

    # factor3 follows xyz order in CloudVolume
    factor3 = np.array([
        2**(validate_mip - chunk_mip), 
        2**(validate_mip - chunk_mip), 1], dtype=np.int32)
    clamped_offset = tuple(go + f - (go - vo) % f for go, vo, f in zip(
        global_offset[::-1], vol.voxel_offset, factor3))
    clamped_stop = tuple(go + s - (go + s - vo) % f
                         for go, s, vo, f in zip(
                             global_offset[::-1], chunk.shape[::-1],
                             vol.voxel_offset, factor3))
    clamped_slices = tuple(
        slice(o, s) for o, s in zip(clamped_offset, clamped_stop))
    clamped_bbox = Bbox.from_slices(clamped_slices)
    clamped_input = chunk.cutout(clamped_slices[::-1])
    # transform to xyz order
    clamped_input = np.transpose(clamped_input)
    # get the corresponding bounding box for validation
    validate_bbox = vol.bbox_to_mip(clamped_bbox, mip=chunk_mip, to_mip=validate_mip)
    #validate_bbox = clamped_bbox // factor3

    # downsample the input using avaraging
    # keep the z as it is since the mip only applies to xy plane
    # recursivly downsample the input
    # if we do it directly, the downsampled input will not be the same with the recursive one
    # because of the rounding error of integer division
    for _ in range(validate_mip - chunk_mip):
        clamped_input = downsample_with_averaging(
            clamped_input, np.array([2, 2, 1], dtype=np.int32))

    # validation by template matching
    assert validate_by_template_matching(clamped_input)

    validate_vol = CloudVolume(
        volume_path,
        bounded=False,
        fill_missing=fill_missing,
        progress=show_progress,
        mip=validate_mip,
        parallel=False)
    validate_input = validate_vol[validate_bbox.to_slices()]
    assert validate_input.shape[3] == 1
    validate_input = np.squeeze(validate_input, axis=3)

    # use the validate input to check the downloaded input
    assert np.alltrue(validate_input == clamped_input)

