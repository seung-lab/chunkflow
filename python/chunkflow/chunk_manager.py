#!/usr/bin/env python
"""@__doc__
manage flowing/sharing chunks that were donated and received
"""

from enum import IntEnum
import numpy as np
from offset_array import OffsetArray
from cloudvolume.storage import Storage
from cloudvolume.cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
import itertools


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
        self.role_mask = OffsetArray(np.zeros((3, 3, 3), dtype=np.int),
                                     global_offset=(-1, -1, -1))
        self._universal_setup()

    def _universal_setup(self):
        for z, y, x in itertools.product(range(-1, 2), range(-1, 2),
                                         range(-1, 2)):
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

    def get_donation_chunk_bbox_list(self, output_block_slices, overlap):
        slices_list, _, _, _ = self._get_chunk_bbox_list_with_role(
            output_block_slices, overlap, Role.Donor)
        return slices_list

    def get_receiving_slices_dict(self, output_block_slices, overlap):
        def _get_slice_list(coordinate, block_slice, size):
            slice_list = [block_slice]
            if coordinate != 0:
                # an edge or corner
                neighboring_slice = slice(block_slice.start +
                                          coordinate * size,
                                          block_slice.stop + coordinate * size)
                slice_list.append(neighboring_slice)
            return slice_list

        ret = dict()
        output_block_size = (s.stop - s.start for s in output_block_slices)
        chunk_slices_list, Z, Y, X = self._get_chunk_bbox_list_with_role(
            output_block_slices, overlap, Role.Receiver)
        for chunk_slices, z, y, x in zip(chunk_slices_list, Z, Y, X):
            chunk_bbox = Bbox.from_slices(chunk_slices)
            chunk_filename = chunk_bbox.to_filename()
            # block slice
            slice_list_z = _get_slice_list(z, output_block_slices[0],
                                           output_block_size[0])
            slice_list_y = _get_slice_list(y, output_block_slices[1],
                                           output_block_size[1])
            slice_list_x = _get_slice_list(x, output_block_slices[2],
                                           output_block_size[2])
            filename_list = list()
            for slicez, slicey, slicex in slice_list_z, slice_list_y, \
                    slice_list_x:
                block_bbox = Bbox.from_slices((slicez, slicey, slicex))
                filename = block_bbox.to_filename() + '%' + chunk_filename
                filename_list.append(filename)

            ret[chunk_slices] = filename_list
        return ret

    def _get_chunk_bbox_list_with_role(self, output_block_slices,
                                       overlap, role):
        """
        get list of donated chunk slices
        args:
            output_block_slices (tuple of slice): slices of output block \
                aligned with cloudvolume backends.
            overlap (tuple of int): the margin size of patch blending
        """
        slices_list = []
        Z, Y, X = self.role_mask.where(self.role_mask == role)

        for z, y, x in zip(Z, Y, X):
            zslice = self._get_chunk_slice(z, output_block_slices[0],
                                           overlap[0])
            yslice = self._get_chunk_slice(y, output_block_slices[1],
                                           overlap[1])
            xslice = self._get_chunk_slice(x, output_block_slices[2],
                                           overlap[2])
            slices_list.append((zslice, yslice, xslice))
        return slices_list, Z, Y, X

    def _get_chunk_slice(self, coordinate, block_slice, margin):
        if coordinate == -1:
            return slice(block_slice.start,
                         block_slice.start + margin)
        elif coordinate == 0:
            return slice(block_slice.start + margin,
                         block_slice.stop)
        elif coordinate == 1:
            return slice(block_slice.stop,
                         block_slice.stop + margin)
        else:
            raise ValueError()


class ChunkManager(object):
    def __init__(self, buffer_array, output_volume, exchange_storage,
                 output_block_slices, overlap, role_mask=RoleMask()):
        """
        Args:
            buffer_array (OffsetArray): the source of donation and \
                destination of receiving, which should be in memory or \
                local disk.
            output_volume (CloudVolume): remote cloud storage or on local \
                bucket.
            exchange_storage (dict or CloudVolume Storage): key value store \
                to host shared chunks which should implement the \
                __setitem__(key, value) function.
            role_mask (3x3x3 OffsetArray): representing the role of each \
                face/edge/corner.
            output_block_slices (tuple of slices): bounding box of aligned \
                output block
            overlap (tuple of int): the margin size of blending region.
        """
        self.buffer_array = buffer_array
        self.output_volume = output_volume
        self.exchange_storage = exchange_storage
        self.role_mask = role_mask
        self.output_block_bbox = Bbox.from_slices(output_block_slices)
        self.overlap = overlap

    def _check(self):
        assert isinstance(self.buffer_array, CloudVolume) or \
            isinstance(self.buffer_array, OffsetArray)
        assert isinstance(self.exchange_storage, dict) or \
            isinstance(self.exchange_storage, Storage)

    def receive_and_blend(self):
        """
        receive chunk and blend.
        assumed that the blending operation is summation.
        """
        # receive and blend
        slices_dict = self.role_mask.get_receiving_slices_dict(
            self.output_block_slices, self.overlap)
        for slices, filename_list in slices_dict:
            chunks = self.exchange_storage.get_files(filename_list)
            for chunk in chunks:
                self.buffer_array[slices] += chunk

        # write blended results, note that this is not aligned
        for slices, filename_list in slices_dict:
            self.output_volume[slices] = self.output_buffer[slices]

    def donate(self):
        """
        donate overlapping chunks to exchanging storage for future blending.
        """
        print("start donating...")
        donor_chunk_slices_list = self.role_mask.get_donation_chunk_bbox_list(
                self.output_block_bbox.to_slices(), self.overlap)
        for chunk_slices in donor_chunk_slices_list:
            bbox = Bbox.from_slices(chunk_slices)
            key = self.output_block_bbox.to_filename() + \
                '%' + bbox.to_filename()
            assert np.any(self.buffer_array.cutout(chunk_slices) > 0)
            self.exchange_storage.put_file(
                key, self.buffer_array.cutout(chunk_slices))

    def save_valid(self):
        """
        save the central valid chunk which do not need to blend with \
            surrounding blocks anymore. The data should be saved to \
            output_volume, which is cloud storage or local bucket.
        """
        print("start saving valid...")
        valid_slices = tuple(slice(o.start+v, o.stop) for o, v in
                             zip(self.output_block_bbox.to_slices(),
                                 self.overlap))
        # this is non-aligned writting, will enhance it for alignment
        self.output_volume[valid_slices] = \
            self.buffer_array.cutout(valid_slices)


if __name__ == '__main__':
    pass
