import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox


def mask(chunk, volume_path, mask_mip, inverse, chunk_mip, 
         fill_missing=False, show_progress=False):
    chunk_bbox = Bbox.from_slices(chunk.slices)
    mask_in_high_mip= _read_mask(volume_path, mask_mip, chunk_mip,
                                 chunk_bbox, inverse=inverse, 
                                 fill_missing=False, show_progress=False)
    if np.alltrue(mask_in_high_mip == 0):
        print('the mask is all black, mask all the voxels directly')
        chunk = 0
        return chunk 
    if np.all(mask_in_high_mip):
        print("mask elements are all positive, return directly")
        return chunk
    if np.alltrue(chunk==0):
        print("chunk is all black, return directly")
        return chunk

    # print("perform masking ...")
    assert np.any(mask_in_high_mip)
    
    # make it the same type with input 
    mask_in_high_mip = mask_in_high_mip.astype(chunk.dtype)
    
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

def _read_mask(mask_volume_path, mask_mip, chunk_mip, chunk_bbox, 
               inverse=True, fill_missing=False, show_progress=False):
    """
    chunk_bbox: the bounding box of the chunk in lower mip level
    inverse_mask: (bool) whether inverse the mask or not
    """
    if not mask_volume_path:
        print('no mask layer path defined')
        return None
    
    # print("download mask chunk...")
    vol = CloudVolume(
        mask_volume_path,
        bounded=False,
        fill_missing=fill_missing,
        progress=show_progress,
        mip=mask_mip)
    # assume that input mip is the same with output mip
    xyfactor = 2**(mask_mip - chunk_mip)
    # only scale the indices in XY plane
    mask_slices = tuple(
        slice(a.start // xyfactor, a.stop // xyfactor)
        for a in bbox.to_slices()[1:3])
    mask_slices = (bbox.to_slices()[0], ) + mask_slices

    # the slices did not contain the channel dimension 
    mask = vol[mask_slices[::-1]]
    mask = np.transpose(mask)
    mask = np.squeeze(mask, axis=0)
    
    if inverse:
        mask = (mask==0)
    return mask

