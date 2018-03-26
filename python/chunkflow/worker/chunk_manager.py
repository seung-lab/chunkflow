#!/usr/bin/env python
"""@__doc__
manage flowing/sharing chunks that were donated and received
"""

from enum import IntEnum
import numpy as np
from .offset_array import OffsetArray
from cloudvolume.storage import Storage
from cloudvolume.lib import Bbox


class Role(IntEnum):
    Donor = 1
    Waster = 0
    Receiver = -1
    # partially donation and partially wasting
    DonorWaster = 2


class ActorTemplate(OffsetArray):
    def __init__(self, role_mask):
        """
        args:
            role_mask (dict): describe the role of each face, whether it is
            receiptor, donor or waster.
        example:
            role_mask = dict(zs=-1, ze=1, ys=0, ye=1, xs=1, xe=-1)
        """
        self._setup_role_mask(role_mask)

        # setup template, which is a 3x3x3 chunk, each voxel except the central
        # one represent a donation/receiving/waster role
        self.role_mask = OffsetArray(np.zeros((3, 3, 3), dtype='np.int'),
                                     offset=(-1, -1, -1))
        self._setup_faces()
        self._setup_edges()
        self._setup_cornors()

        # use only Donor and Receiptor for now, the implementation is easier
        self.role_mask[self.role_mask >= 0] = Role.Donor
        self.role_mask[0, 0, 0] = Role.Waster

    def _setup_role_mask(self, role_mask):
        if isinstance(role_mask, list) or isinstance(role_mask, np.ndarray):
            self.role_mask = dict(zs=role_mask[0],
                                  ze=role_mask[1],
                                  ys=role_mask[2],
                                  ye=role_mask[3],
                                  xs=role_mask[4],
                                  xe=role_mask[5])
        else:
            assert isinstance(role_mask, dict)
            self.role_mask = role_mask

    def _setup_faces(self):
        self.role_mask[-1, 0,  0] = self.role_mask['zs']
        self.role_mask[1,  0,  0] = self.role_mask['ze']
        self.role_mask[0, -1,  0] = self.role_mask['ys']
        self.role_mask[0,  1,  0] = self.role_mask['ye']
        self.role_mask[0,  0, -1] = self.role_mask['xs']
        self.role_mask[0,  0,  1] = self.role_mask['xe']

    def _setup_edges(self):
        def _get_edge_role(r1, r0):
            if Role.Donor == r1 and Role.Donor == r0:
                return Role.Donor
            elif Role.Waster == r1 and Role.Waster == r0:
                return Role.Waster
            elif (r1 == Role.Waster and r0 == Role.Donor) or \
                 (r0 == Role.Waster and r1 == Role.Donor):
                return Role.DonorWaster
            elif r1 == Role.Receiver or r0 == Role.Receiver:
                return Role.Receiver
            else:
                raise NotImplementedError()

        # y,x
        self.role_mask[0, -1, -1] = _get_edge_role(self.role_mask['xs'],
                                                   self.role_mask['ys'])
        self.role_mask[0, -1,  1] = _get_edge_role(self.role_mask['xe'],
                                                   self.role_mask['ys'])
        self.role_mask[0,  1, -1] = _get_edge_role(self.role_mask['xs'],
                                                   self.role_mask['ye'])
        self.role_mask[0,  1,  1] = _get_edge_role(self.role_mask['xe'],
                                                   self.role_mask['ye'])
        # z,x
        self.role_mask[-1, 0, -1] = _get_edge_role(self.role_mask['xs'],
                                                   self.role_mask['zs'])
        self.role_mask[-1, 0,  1] = _get_edge_role(self.role_mask['xe'],
                                                   self.role_mask['zs'])
        self.role_mask[1,  0, -1] = _get_edge_role(self.role_mask['xs'],
                                                   self.role_mask['ze'])
        self.role_mask[1,  0,  1] = _get_edge_role(self.role_mask['xe'],
                                                   self.role_mask['ze'])
        # z,y
        self.role_mask[-1, -1, 0] = _get_edge_role(self.role_mask['ys'],
                                                   self.role_mask['zs'])
        self.role_mask[-1, 1,  0] = _get_edge_role(self.role_mask['ye'],
                                                   self.role_mask['zs'])
        self.role_mask[1, -1,  0] = _get_edge_role(self.role_mask['ys'],
                                                   self.role_mask['ze'])
        self.role_mask[1,  1,  0] = _get_edge_role(self.role_mask['ye'],
                                                   self.role_mask['ze'])

    def _setup_cornors(self):
        def _get_cornor_role(r0, r1, r2):
            if Role.Donor == r0 and Role.Donor == r1 and Role.Donor == r2:
                return Role.Donor
            elif Role.Receiver in (r0, r1, r2):
                return Role.Receiver
            elif (Role.Donor in (r0, r1, r2)) \
                    and (Role.Waster in (r0, r1, r2)) \
                    and (Role.Receiver not in (r0, r1, r2)):
                return Role.DonorWaster
            else:
                NotImplementedError()

        self.role_mask[-1, -1, -1] = _get_cornor_role(self.role_mask['zs'],
                                                      self.role_mask['ys'],
                                                      self.role_mask['xs'])
        self.role_mask[-1, -1,  1] = _get_cornor_role(self.role_mask['zs'],
                                                      self.role_mask['ys'],
                                                      self.role_mask['xe'])
        self.role_mask[-1,  1,  -1] = _get_cornor_role(self.role_mask['zs'],
                                                       self.role_mask['ye'],
                                                       self.role_mask['xs'])
        self.role_mask[-1,  1,   1] = _get_cornor_role(self.role_mask['zs'],
                                                       self.role_mask['ye'],
                                                       self.role_mask['xe'])
        self.role_mask[1,  -1,  -1] = _get_cornor_role(self.role_mask['ze'],
                                                       self.role_mask['ys'],
                                                       self.role_mask['xs'])
        self.role_mask[1,  -1,   1] = _get_cornor_role(self.role_mask['ze'],
                                                       self.role_mask['ys'],
                                                       self.role_mask['xe'])
        self.role_mask[1,   1,  -1] = _get_cornor_role(self.role_mask['ze'],
                                                       self.role_mask['ye'],
                                                       self.role_mask['xs'])
        self.role_mask[1,   1,   1] = _get_cornor_role(self.role_mask['ze'],
                                                       self.role_mask['ye'],
                                                       self.role_mask['xe'])


class ReceiverManager(object):
    """
    receive chunks

    args:
        role_mask (3x3x3 ndarray): the mask indicating faces/edge/corner role
        output_bbox (tuple(slicez,slicey,slicex)): the slice of output
        shared_margin_size (tuple(Z,Y,X)): the margin size received
    """
    def __init__(self, exchange_url, buffer_chunk, role_mask, output_bbox,
                 shared_margin_size, num_threads=8):
        self.exchange_url = exchange_url
        self.role_mask = role_mask
        self.output_bbox = output_bbox
        self.shared_margin_size = shared_margin_size
        self.num_threads = num_threads

        assert isinstance(output_bbox, Bbox)

        self._setup_receiving_chunk_bbox_list()

    def _setup_receiving_chunk_bbox_list(self):
        self.receiving_chunk_bbox_list = []
        Z, Y, X = self.role_mask.where(self.role_mask == Role.Receiver)
        for z, y, x in zip(Z, Y, X):
            receiving_bbox = Bbox.from_slices(
                self._get_receiving_slice(c, o, s) for c, o, s in zip(
                    [z, y, x], self.output_bbox, self.shared_margin_size))
            self.receiving_chunk_bbox_list.append(receiving_bbox)

    @property
    def key_list(self):
        key_list = []
        for chunk_bbox in self.receiving_chunk_bbox_list:
            key = chunk_bbox.to_filename()
            key_list.append(key)
        return key_list

    def receive_and_blend(self):
        self._receive_chunks()
        for chunk in self.chunk_list:
            self.buffer_chunk.add_overlap(chunk)

    def _receive_chunks(self):
        self.chunk_list = []
        with Storage(self.exchange_url, num_threads=self.num_threads) as s:
            results = s.get_files(self.key_list)
            for i, key in enumerate(self.key_list):
                array = results[i]['content']
                # make sure that all the required chunks were received
                assert array is not None
                bbox = Bbox.from_filename(key)
                chunk = OffsetArray.from_bbox(array, bbox)
                self.chunk_list.append(chunk)

    def _get_receiving_slice(self, coordinate, output_slice, shared_margin):
        if coordinate == -1:
            return slice(output_slice.start,
                         output_slice.start + shared_margin)
        elif coordinate == 0:
            return slice(output_slice.start + shared_margin,
                         output_slice.stop - shared_margin)
        elif coordinate == 1:
                return slice(output_slice.stop - shared_margin,
                             output_slice.stop)
        else:
            raise ValueError('invalid coordinate')


class DonorManager(object):
    def __init__(self, exchange_url, buffer_chunk, role_mask, output_bbox,
                 shared_margin_size, num_threads=8):
        self.exchange_url = exchange_url
        self.buffer_chunk = buffer_chunk
        self.role_mask = role_mask
        self.output_bbox = output_bbox
        self.shared_margin_size = shared_margin_size
        self.num_threads = num_threads

        assert isinstance(buffer_chunk, OffsetArray)

    @property
    def chunk_bbox_list(self):
        """
        decompose to small chunks with size of half of a patch size
        check all the decomposed chunk bbox, remove the ones completly
        inside output slice.
        """
        chunk_bbox_list = []
        Z, Y, X = self.role_mask.where(self.role_mask == Role.Donor)
        for z, y, x in zip(Z, Y, X):
            bbox_list_z = self._get_chunk_bbox_list(
                z, self.output_slice[0], self.shared_margin_size[0])
            bbox_list_y = self._get_chunk_bbox_list(
                y, self.output_slice[1], self.shared_margin_size[1])
            bbox_list_x = self._get_chunk_bbox_list(
                x, self.output_slice[2], self.shared_margin_size[2])
            # this candidate bbox list contains all the decomposed donating
            # patches
            candidate_bbox_list = [(rz, ry, rx)
                                   for rz in bbox_list_z
                                   for ry in bbox_list_y
                                   for rx in bbox_list_x]
            # filter out the chunk which were completly inside the output
            num_inside = 0
            for bbox in candidate_bbox_list:
                # check whether the bbox is completly inside the output
                # bbox or not
                if np.all(ri.start >= ro.start and ri.stop <= ro.stop
                          for ri, ro in zip(bbox, self.output_bbox)):
                    # this is a chunk used by itself, will not donating out
                    num_inside += 1
                else:
                    chunk_bbox_list.append(bbox)
            # only one of them should be found here
            assert num_inside == 1
        return chunk_bbox_list

    def donate(self):
        """
        donate the chunks to exchange place
        """
        with Storage(self.exchange_url, n_threads=self.num_threads) as s:
            file_list = []
            for chunk_bbox in self.chunk_bbox_list:
                chunk = self.buffer_chunk[chunk_bbox.to_slices()]
                key = chunk_bbox.to_filename()
                file_list.append((key, chunk))
            s.put_files(file_list, compress='gzip', block=True)

    def _get_chunk_bbox_list(self, coordinate, output_slice,
                             shared_margin):
        half_patch_bbox_list = []
        if coordinate == -1:
            r = slice(output_slice.start - shared_margin,
                      output_slice.start)
            half_patch_bbox_list.append(r)
            r = slice(output_slice.start,
                      output_slice.start + shared_margin)
            half_patch_bbox_list.append(r)
        elif coordinate == 0:
            r = slice(output_slice.start + shared_margin,
                      output_slice.stop - shared_margin)
            half_patch_bbox_list.append(r)
        elif coordinate == 1:
            r = slice(output_slice.stop - shared_margin,
                      output_slice.stop)
            half_patch_bbox_list.append(r)
            r = slice(output_slice.stop,
                      output_slice.stop + shared_margin)
            half_patch_bbox_list.append(r)
        else:
            raise ValueError('invalid coordinate')

        return half_patch_bbox_list


class ChunkManager(object):
    def __init__(self, exchange_url, buffer_chunk, role_mask,
                 output_bbox, shared_margin_size):
        self.exchange_url = exchange_url
        self.buffer_chunk = buffer_chunk
        self.role_mask = ActorTemplate(role_mask)
        self.output_bbox = output_bbox
        self.shared_margin_size = shared_margin_size

        # initialized as no dependent chunks
        self.receiving_chunk_bbox_list = []
        self.donating_chunk_bbox_list = []
        self.input_chunk_bbox = [slice(o.start-m, o.stop+m)
                                 for (o, m) in zip(self.output_bbox,
                                                   self.shared_margin_size)]

    def receive_and_blend(self):
        receiver_manager = ReceiverManager(self.exchange_url,
                                           self.buffer_chunk, self.role_mask,
                                           self.output_bbox,
                                           self.shared_margin_size)
        receiver_manager.receive_and_blend()

    def donate(self):
        donate_manager = DonorManager(self.exchange_url,
                                      self.buffer_chunk, self.role_mask,
                                      self.output_bbox,
                                      self.shared_margin_size)
        donate_manager.donate()


if __name__ == '__main__':
    pass
