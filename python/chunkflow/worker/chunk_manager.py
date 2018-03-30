#!/usr/bin/env python
"""@__doc__
manage flowing/sharing chunks that were donated and received
"""

from enum import IntEnum
import numpy as np
from .offset_array import OffsetArray
from cloudvolume.storage import Storage
from cloudvolume.cloudvolume import CloudVolume
from cloudvolume.lib import Bbox


class Role(IntEnum):
    # PoorGuy = -2    # not receiving anything
    Receiver = -1
    NULL = 0        # used for central voxel
    Donor = 1
    # Waster = 2


class RoleMask(object):
    def __init__(self):
        # setup template, which is a 3x3x3 chunk, each voxel except the central
        # one represent a donation/receiving/waster role
        self.role_mask = OffsetArray(np.zeros((3, 3, 3), dtype='np.int'),
                                     offset=(-1, -1, -1))
        self._universal_setup()

    def _universal_setup(self):
        for z, y, x in zip(range(-1, 2), range(-1, 2), range(-1, 2)):
            if x+y+z > 0:
                self.role_mask[z, y, x] = Role.Donor
            elif x+y+z < 0:
                self.role_mask[z, y, x] = Role.Receiver
            elif x == 0 and y == 0 and z == 0:
                pass
            else:
                # one zero, one -1, one 1
                # if the first nonzero value is positive then donor,
                # otherwise it is receiver. this will balance the
                # number
                for k in [z, y, x]:
                    if k != 0:
                        if k < 0:
                            self.role_mask[z, y, x] = Role.Receiver
                        else:
                            self.role_mask[z, y, x] = Role.Donor

    def get_donation_chunk_bbox_list(self, output_block_slices, blending_size):
        return self._get_chunk_bbox_list_with_role(output_block_slices,
                                                   blending_size, Role.Donor)

    def get_receiving_chunk_bbox_list(self, output_block_slices,
                                      blending_size):
        return self._get_chunk_bbox_list_with_role(output_block_slices,
                                                   blending_size, Role.Donor)

    def _get_chunk_bbox_list_with_role(self, output_block_slices,
                                       blending_size, role):
        """
        get list of donated chunk slices
        args:
            output_block_slices (tuple of slice): slices of output block \
                aligned with cloudvolume backends.
            blending_size (tuple of int): the margin size of patch blending
        """
        donation_slices_list = []
        Z, Y, X = self.role_mask.where(self.role_mask == role)

        for z, y, x in zip(Z, Y, X):
            zslice = self._get_chunk_slice(z, output_block_slices[0],
                                           blending_size[0])
            yslice = self._get_chunk_slice(y, output_block_slices[1],
                                           blending_size[1])
            xslice = self._get_chunk_slice(x, output_block_slices[2],
                                           blending_size[2])
            bbox = Bbox.from_slices((zslice, yslice, xslice))
            donation_slices_list.append(bbox)

    def _get_chunk_slice(self, coordinate, output_block_slice,
                         blending_margin):
        if coordinate == -1:
            return slice(output_block_slice.start,
                         output_block_slice.start + blending_margin)
        elif coordinate == 0:
            return slice(output_block_slice.start + blending_margin,
                         output_block_slice.stop)
        elif coordinate == 1:
            return slice(output_block_slice.stop,
                         output_block_slice.stop + blending_margin)
        else:
            raise ValueError()


class ChunkManager(object):
    def __init__(self, buffer_volume, exchange_storage,
                 output_block_bbox, blending_size, role_mask=RoleMask()):
        """
        Args:
            buffer_volume (CloudVolume or OffsetArray): the source of \
                donation and destination of receiving
            exchange_storage (dict or CloudVolume Storage): key value store \
                to host shared chunks
            role_mask (3x3x3 OffsetArray): representing the role of each \
                face/edge/corner.
            output_block_bbox (Bbox): bounding box of aligned output block
            blending_size (tuple of int): the margin size of blending region.
        """
        self.buffer_volume = buffer_volume
        self.exchange_storage = exchange_storage
        self.role_mask = role_mask
        self.output_block_bbox = output_block_bbox
        self.blending_size = blending_size

    def _check(self):
        assert isinstance(self.buffer_volume, CloudVolume) or \
            isinstance(self.buffer_volume, OffsetArray)
        assert isinstance(self.exchange_storage, dict) or \
            isinstance(self.exchange_storage, Storage)

    def receive_and_blend(self):
        """
        receive chunk and blend.
        assumed that the blending operation is summation.
        """
        chunk_slices_list = self.role_mask.get_receiving_chunk_bbox_list(
                self.output_block_bbox, self.blending_size)
        for chunk_slices in chunk_slices_list:
            bbox = Bbox.from_slices(chunk_slices)
            self.buffer_volume[chunk_slices] += \
                self.exchange_storage[bbox.to_filename()]

    def donate(self):
        """
        donate chunks to exchanging storage
        """
        donor_chunk_slices_list = self.role_mask.get_donation_chunk_bbox_list(
                self.output_block_bbox, self.blending_size)
        for chunk_slices in donor_chunk_slices_list:
            bbox = Bbox.from_slices(chunk_slices)
            self.exchange_storage[bbox.to_filename()] = \
                self.buffer_volume[chunk_slices]


if __name__ == '__main__':
    pass
