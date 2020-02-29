from warnings import warn
import numpy as np

from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox

from chunkflow.chunk import Chunk
from .base import OperatorBase


class MaskOperator(OperatorBase):
    def __init__(self,
                 volume_path: str,
                 mask_mip: int,
                 chunk_mip: int,
                 inverse: bool = False,
                 fill_missing: bool = False,
                 check_all_zero=False,
                 verbose: int = 1,
                 name: str = 'mask'):
        super().__init__(name=name, verbose=verbose)

        self.mask_mip = mask_mip
        self.chunk_mip = chunk_mip
        self.inverse = inverse
        self.volume_path = volume_path
        self.check_all_zero = check_all_zero

        self.mask_vol = CloudVolume(volume_path,
                                    bounded=False,
                                    fill_missing=fill_missing,
                                    progress=verbose,
                                    mip=mask_mip)

        if verbose:
            print(f'build mask operator based on {volume_path} at mip {mask_mip}')

    def __call__(self, x):
        if self.check_all_zero:
            assert isinstance(x, Bbox)
            return self.is_all_zero(x)
        else:
            assert isinstance(x, Chunk)
            return self.maskout(x)

    def is_all_zero(self, bbox):
        mask_in_high_mip = self._read_mask_in_high_mip(bbox)
        # To-Do: replace with np.array_equiv function
        # return np.array_equiv(mask_in_high_mip, 0)
        return np.alltrue(mask_in_high_mip == 0)

    def maskout(self, chunk):
        if self.verbose:
            print('mask out chunk using {} in mip {}'.format(
                self.volume_path, self.mask_mip))
        
        if np.alltrue(chunk == 0):
            warn("chunk is all black, return directly")
            return chunk

        chunk_bbox = Bbox.from_slices(chunk.slices[-3:])
        mask_in_high_mip = self._read_mask_in_high_mip(chunk_bbox)

        if np.alltrue(mask_in_high_mip == 0):
            warn('the mask is all black, mask all the voxels directly')
            np.multiply(chunk, 0, out=chunk)
            return chunk
        if np.all(mask_in_high_mip):
            warn("mask elements are all positive, return directly")
            return chunk

        assert np.any(mask_in_high_mip)

        # make it the same type with input
        mask_in_high_mip = mask_in_high_mip.astype(chunk.dtype)
        
        # upsampling factor in XY plane
        #mask = np.zeros(chunk.shape[-3:], dtype=chunk.dtype)
        xyfactor = 2**(self.mask_mip - self.chunk_mip)
        for offset in np.ndindex((xyfactor, xyfactor)):
            chunk.array[..., 
                        np.s_[offset[0]::xyfactor], 
                        np.s_[offset[1]::xyfactor]] *= mask_in_high_mip
        #    mask[:, 
        #         np.s_[offset[0]::xyfactor], 
        #         np.s_[offset[1]::xyfactor]] = mask_in_high_mip

        #if chunk.ndim == mask.ndim:
        #    np.multiply(chunk, mask, out=chunk)
        #elif chunk.ndim == mask.ndim + 1:
        #    for channel in range(chunk.shape[0]):
        #        np.multiply(chunk[channel, :, :, :],
        #                    mask,
        #                    out=chunk[channel, :, :, :])
        #else:
        #    raise ValueError('invalid chunk or mask dimension.')
        return chunk

    def _read_mask_in_high_mip(self, chunk_bbox):
        """
        chunk_bbox: the bounding box of the chunk in lower mip level
        """
        # print("download mask chunk...")
        # make sure that the slices only contains zyx without channel
        chunk_slices = chunk_bbox.to_slices()[-3:]
        chunk_size = chunk_bbox.maxpt - chunk_bbox.minpt

        # assume that input mip is the same with output mip
        xyfactor = 2**(self.mask_mip - self.chunk_mip)
        # only scale the indices in XY plane
        # only scale the indices in XY plane
        mask_slices = tuple(
            slice(a.start // xyfactor, a.stop // xyfactor)
            for a in chunk_slices[1:3])

        # we have to make sure that chunk size is divisible by xyfactor
        # so the rounding is not neccesary. 
        #def _round_div(n, d):
        #    return (n + d // 2) // d
        #
        #high_mip_xysize = (_round_div(s, xyfactor) for s in chunk_size[-2:]) 

        #mask_slices = tuple(
        #    slice(_round_div(a.start, xyfactor), 
        #          _round_div(a.start, xyfactor) + s) 
        #    for a, s in zip(chunk_slices[-2:], high_mip_xysize))
        
        mask_slices = (chunk_slices[-3], ) + mask_slices
        
        # the slices did not contain the channel dimension
        mask = self.mask_vol[mask_slices[::-1]]
        # this is a cloudvolume VolumeCutout rather than a normal numpy array
        # which will make np.alltrue(mask_in_high_mip == 0) to be
        # VolumeCutout(False) rather than False, so we need to transform it 
        # to numpy
        mask = mask.astype(np.bool)
        mask = np.asarray(mask)
        mask = np.transpose(mask)
        mask = np.squeeze(mask, axis=0)

        if self.inverse:
            mask = (mask == 0)
        return mask
