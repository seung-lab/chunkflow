from warnings import warn
import numpy as np

from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox

from .operator_base import OperatorBase


class MaskOperator(OperatorBase):
    def __init__(self, volume_path: str, mask_mip: int, chunk_mip: int, 
                 inverse: bool=False, fill_missing: bool=False, 
                 verbose: bool=True, name: str='mask'):
        super().__init__(name=name)

        self.mask_mip = mask_mip
        self.chunk_mip = chunk_mip
        self.inverse = inverse
        self.verbose = verbose
        self.mask_vol = CloudVolume(
            volume_path,
            bounded=False,
            fill_missing=fill_missing,
            progress=verbose,
            mip=mask_mip)
        
        if verbose:
            print("mask chunk using {} at mip {}".format(
                volume_path, mask_mip))

    def __call__(self, chunk):
        chunk_bbox = Bbox.from_slices(chunk.slices[-3:])
        mask_in_high_mip = self._read_mask(chunk_bbox)
        # this is a cloudvolume VolumeCutout rather than a normal numpy array
        # which will make np.alltrue(mask_in_high_mip == 0) to be 
        # VolumeCutout(False) rather than False
        mask_in_high_mip = np.asarray(mask_in_high_mip)
        if self.inverse:
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
        
        assert np.any(mask_in_high_mip)
        
        # make it the same type with input 
        mask_in_high_mip = mask_in_high_mip.astype(chunk.dtype)
       
        if self.verbose:
            print("upsampling mask ...")
        # upsampling factor in XY plane
        mask = np.zeros(chunk.shape[-3:], dtype=chunk.dtype)
        xyfactor = 2**(self.mask_mip - self.chunk_mip)
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

    def _read_mask(self, chunk_bbox):
        """
        chunk_bbox: the bounding box of the chunk in lower mip level
        """
        # print("download mask chunk...")
        # make sure that the slices only contains zyx without channel
        chunk_slices = chunk_bbox.to_slices()[-3:]
        # assume that input mip is the same with output mip
        xyfactor = 2**(self.mask_mip - self.chunk_mip)
        # only scale the indices in XY plane
        mask_slices = tuple(
            slice(a.start // xyfactor, a.stop // xyfactor)
            for a in chunk_slices[1:3])
        mask_slices = (chunk_slices[0], ) + mask_slices

        # the slices did not contain the channel dimension 
        mask = self.mask_vol[mask_slices[::-1]]
        mask = np.transpose(mask)
        mask = np.squeeze(mask, axis=0)
        
        return mask

