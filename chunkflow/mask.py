import numpy as np
from warnings import warn

from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
from .lib.offset_array import OffsetArray


def mask(chunk, volume_path, mask_mip, chunk_mip, 
         inverse=False, fill_missing=False, verbose=True):
    chunk_bbox = Bbox.from_slices(chunk.slices[-3:])
    mask_in_high_mip = _read_mask(volume_path, mask_mip, chunk_mip,
                                 chunk_bbox,  
                                 fill_missing=fill_missing, verbose=verbose)
    # this is a cloudvolume VolumeCutout rather than a normal numpy array
    # which will make np.alltrue(mask_in_high_mip == 0) to be VolumeCutout(False) 
    # rather than False
    mask_in_high_mip = np.asarray(mask_in_high_mip)
    if inverse:
        mask_in_high_mip = (mask_in_high_mip==0)

    if np.alltrue(mask_in_high_mip == 0):
        warn('the mask is all black, mask all the voxels directly')
        np.multiply(chunk, 0, out=chunk)
        return chunk
    if np.all(mask_in_high_mip):
        warn("mask elements are all positive, return directly")
        return chunk
    if np.alltrue(chunk==0):
        warn("chunk is all black, return directly")
        return chunk
    
    if verbose:
        print("mask chunk using {} at mip {}".format(volume_path, mask_mip))
    assert np.any(mask_in_high_mip)
    
    # make it the same type with input 
    mask_in_high_mip = mask_in_high_mip.astype(chunk.dtype)
   
    if verbose:
        print("upsampling mask ...")
    # upsampling factor in XY plane
    mask = np.zeros(chunk.shape[-3:], dtype=chunk.dtype)
    xyfactor = 2**(mask_mip - chunk_mip)
    for offset in np.ndindex((xyfactor, xyfactor)):
        mask[:, 
             np.s_[offset[0]::xyfactor], 
             np.s_[offset[1]::xyfactor]] = mask_in_high_mip

    if chunk.ndim == mask.ndim:
        np.multiply(chunk, mask, out=chunk)
    elif chunk.ndim == mask.ndim + 1:
        for channel in range(chunk.shape[0]):
            np.multiply(chunk[channel, :, :, :], mask, 
                        out=chunk[channel, :, :, :])
    else:
        raise ValueError('invalid chunk or mask dimension.')
    return chunk

def _read_mask(mask_volume_path, mask_mip, chunk_mip, chunk_bbox, 
               fill_missing=False, verbose=True):
    """
    chunk_bbox: the bounding box of the chunk in lower mip level
    """
    if not mask_volume_path:
        print('no mask layer path defined')
        return None
    
    # print("download mask chunk...")
    # make sure that the slices only contains zyx without channel
    chunk_slices = chunk_bbox.to_slices()[-3:]
    vol = CloudVolume(
        mask_volume_path,
        bounded=False,
        fill_missing=fill_missing,
        progress=verbose,
        mip=mask_mip)
    # assume that input mip is the same with output mip
    xyfactor = 2**(mask_mip - chunk_mip)
    # only scale the indices in XY plane
    mask_slices = tuple(
        slice(a.start // xyfactor, a.stop // xyfactor)
        for a in chunk_slices[1:3])
    mask_slices = (chunk_slices[0], ) + mask_slices

    # the slices did not contain the channel dimension 
    mask = vol[mask_slices[::-1]]
    mask = np.transpose(mask)
    mask = np.squeeze(mask, axis=0)
    
    return mask

