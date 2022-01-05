from __future__ import annotations

import os
import json
from typing import List
from copy import deepcopy

import numpy as np
import h5py

from chunkflow.lib.bounding_boxes import BoundingBox, Cartesian


class Synapses():
    def __init__(self, pre: np.ndarray, pre_confidence: np.ndarray = None, 
            post: np.ndarray = None, post_confidence: np.ndarray = None, 
            resolution: Cartesian = None) -> None:
        """Synapses containing T-bars and post-synapses

        Args:
            pre (np.ndarray): T-bar points, Nx3, z,y,x, the coordinate should be physical coordinate rather than voxels.
            pre_confidence (np.ndarray, optional): confidence of T-bar detection. Defaults to None.
            post (np.ndarray, optional): [description]. Defaults to None.
            resolution (Cartesian, optional): [description]. Defaults to None.
        """
        assert pre.ndim == 2
        assert pre.shape[1] == 3

        if pre_confidence is not None:
            assert pre_confidence.ndim == 1
            np.testing.assert_array_less(pre_confidence, 1.00001)
            np.testing.assert_array_less(-0.0001, pre_confidence)
            assert len(pre_confidence) == pre.shape[0]

        if post is not None:
            if post_confidence is not None:
                assert post_confidence.ndim == 1
                assert len(post_confidence) == post.shape[1]

            assert post.ndim == 2
            # parent pre index, z, y, x
            assert post.shape[1] == 4
            assert np.issubdtype(post.dtype, np.integer)

        if resolution is not None:
            resolution = np.asarray(resolution, dtype=pre.dtype)
            np.testing.assert_array_less(0, resolution)

        self.resolution = resolution
        # unsigned integer will have minus issues
        self.pre = pre.astype(np.int32)
        self.pre_confidence = pre_confidence
        if post is not None:
            post = post.astype(np.int32)
        self.post = post
        self.post_confidence = post_confidence
        
    @classmethod
    def from_dict(cls, dc: dict):
        """Synapses as a dictionary

        Args:
            synapses (dict): the whole synapses in a dictionary
        """
        order = dc['order']
        resolution = dc['resolution']
        del dc['order']
        del dc['resolution']

        pre_num = len(synapses)
        pre = np.zeros((pre_num, 3), dtype=np.int32)
        post_list = []
        pre_indices = []
        for sid, synapse in enumerate(synapses.values()):
            pre[sid, :] = np.asarray(synapse['coord'])
            if 'postsynapses' in synapse:
                for idx, post_coordinate in enumerate(synapse['postsynapses']):
                    post_list.append(post_coordinate)
                    pre_indices.append(sid)
        
        post_num = len(post_list)
        if post_num == 0:
            post = None
        else:
            post = np.zeros((post_num, 4), dtype=np.int32)
            for idx, post_synapse in enumerate(post_list):
                post[idx, 0] = pre_indices[idx]
                post[idx, 1:] = np.asarray(post_synapse)

        if order == ['z', 'y', 'x']:
            pass
        elif order == ['x', 'y', 'z']:
            resolution = resolution[::-1]
            # invert to z,y,x
            pre = np.fliplr(pre)
            if post is not None:
                post[:, 1:] = np.fliplr(post[:, 1:])

        return cls(pre, post=post, resolution=resolution)

    @classmethod
    def from_dvid_list(cls, syns: list, resolution: Cartesian=None):
        """from a dict fetched from DVID using fivol

        Args:
            syns (list): the synapse list fetched from DVID

        Returns:
            Synapses: a Synapses instance

        Example:
            syns = fivol.get_syndata(dvid_url, uuid)
            synapses = Synapses.from_dvid_list(syns)
        """
        pre_list = []
        pre_proofread = []
        post_list = []
        pre_confidence = []
        for syn in syns:
            if 'Pre' in syn['Kind']:
                # map from xyz to zyx
                pos = syn['Pos'][::-1]
                pos = Cartesian(*pos)
                pre_list.append(pos)
                if syn['Prop']['user'] == 'jwu':
                    pre_proofread.append(False)
                else:
                    pre_proofread.append(True)

                if 'conf' in syn['Prop']:
                    conf = syn['Prop']['conf']
                    conf = float(conf)
                else:
                    conf = 0.5
                pre_confidence.append(conf)
            elif 'Post' in syn['Kind']:
                # map from xyz to zyx
                pos = syn['Pos'][::-1]
                pos = Cartesian(*pos)
                # if 'To' not in syn['Prop']:
                    # print(syn)
                    # breakpoint()
                pre_pos = syn['Rels'][0]['To'][::-1]
                pre_pos = Cartesian(*pre_pos)
                post_list.append((pos, pre_pos))
            else:
                raise ValueError('unexpected synapse type: ', syn)
        
        # build a map from pre position to index
        pre_pos2idx = {}
        for idx, pos in enumerate(pre_list):
            pre_pos2idx[pos] = idx
        assert len(pre_pos2idx) == len(pre_list)
        
        post_to_pre_indices = []
        for _, pre_pos in post_list:
            pre_idx = pre_pos2idx[ pre_pos ]
            post_to_pre_indices.append(pre_idx)
        assert len(post_to_pre_indices) == len(post_list)

        pre = np.asarray(pre_list, dtype=np.int32)
        pre_confidence = np.asarray(pre_confidence, dtype=np.float32)
        post_to_pre_indices = np.asarray(post_to_pre_indices, dtype=np.int32)
        post_list = [x[0] for x in post_list]
        post_list = np.asarray(post_list, dtype=np.int32)
        post_to_pre_indices = np.expand_dims(post_to_pre_indices, 1)
        post = np.hstack((post_to_pre_indices, post_list))
        return cls(
            pre, post=post, 
            pre_confidence=pre_confidence,
            resolution=resolution,
        )
            
    @classmethod
    def from_json(cls, fname: str, resolution: tuple = None):
        with open(fname, 'r') as file:
            syns = json.load(file)

        if resolution is not None:
            syns['resolution'] = resolution
        return cls.from_dict(syns)

    @classmethod
    def from_h5(cls, fname: str, resolution: tuple = None, c_order: bool = True):
        with h5py.File(fname, 'r') as hf:
            if 'pre' in hf.keys():
                pre = hf['pre']
            else:
                # this only works with an old version
                # we can delete this code once we get ride of old version
                pre = hf['tbars']
            pre = np.asarray(pre, dtype=np.int32)

            if resolution is None and 'resolution' in hf.keys():
                resolution = np.asarray(hf['resolution'])
            
            if 'post' in hf.keys():
                post = np.asarray(hf['post'], dtype=np.int32)
            else:
                post = None
            
            if 'pre_confidence' in hf.keys():
                pre_confidence = np.asarray(hf['pre_confidence'])
            else:
                pre_confidence = None
            
            if 'post_confidence' in hf.keys():
                post_confidence = np.asarray(hf['post_confidence'])
            else:
                post_confidence = None

        if not c_order:
            # transform to C order
            pre = pre[:, ::-1]
            post[:, 1:] = post[:, 1:][:, ::-1]

        return cls(pre, post=post, pre_confidence=pre_confidence, 
                    post_confidence=post_confidence, resolution=resolution)

    def to_h5(self, fname: str) -> None:
        """save to a HDF5 file

        Args:
            fname (str): the file name to be saved
        """
        assert fname.endswith(".h5") or fname.endswith(".hdf5")
        with h5py.File(fname, "w") as hf:
            
            hf['pre'] = self.pre

            if self.post is not None:
                hf['post'] = self.post
            
            if self.resolution is not None:
                hf['resolution'] = self.resolution
            
            if self.pre_confidence is not None:
                hf['pre_confidence'] = self.pre_confidence

            if self.post_confidence is not None:
                hf['post_confidence'] = self.post_confidence

    @classmethod
    def from_file(cls, fname: str, resolution: tuple = None, c_order: bool = True):
        assert os.path.exists(fname)
        if fname.endswith('.json'):
            assert c_order
            return cls.from_json(fname, resolution = resolution)
        elif fname.endswith('.h5'):
            return cls.from_h5(fname, resolution=resolution, c_order=c_order)
        else:
            raise ValueError(f'only support JSON and HDF5 file, but got {fname}')

    def add_pre(self, pre: np.ndarray, confidence: float = 1.):
        """add some additional pre synapses

        Args:
            pre (np.ndarray): the coordinates of the additional pre synapses. The shape should be Nx3
        """
        assert pre.shape[1] == 3
        assert pre.ndim == 2
        self.pre = np.vstack((self.pre, pre))
        if self.pre_confidence is not None:
            assert confidence >= 0.
            assert confidence <= 1.
            confidences = np.ones((pre.shape[0],), dtype=np.float32) * confidence
            self.pre_confidence = np.concatenate((self.pre_confidence, confidences), axis=None)
        return self

    def __eq__(self, other: Synapses) -> bool:
        """compare two synapses.
        Note that we do not compare the confidence here!

        Args:
            other (Synapses): the other Synapses instance

        Returns:
            bool: whether the pre and post are the same
        """
        if np.array_equal(self.pre, other.pre):
            if self.post is None and other.post is None:
                return True
            elif self.post is not None and other.post is not None and np.array_equal(
                    self.post, other.post):
                return True
            else:
                return False
        else:
            return False

    @property
    def post_coordinates(self) -> np.ndarray:
        """the coordinate array

        Returns:
            np.ndarray: the array of coordinates. for each row, z,y,x
        """
        return self.post[:, 1:]

    @property
    def pre_num(self) -> int:
        return self.pre.shape[0]
    
    @property
    def post_num(self) -> int:
        return self.post.shape[0]

    @property
    def pre_bounding_box(self) -> BoundingBox:
        bbox = BoundingBox.from_points(self.pre)
        # the end point is exclusive
        bbox.adjust((0,0,0, 1,1,1))
        return bbox

    def post_bounding_box(self) -> BoundingBox:
        bbox = BoundingBox.from_points(self.post_coordinates)
        # the right direction is exclusive
        bbox.adjust((0,0,0, 1,1,1))
        return bbox
    
    @property
    def bounding_box(self) -> BoundingBox:
        bbox = self.pre_bounding_box
        bbox.union(self.post_bounding_box)
        return bbox
    
    @property
    def pre_with_physical_coordinate(self) -> np.ndarray:
        if self.resolution is not None:
            return self.pre * self.resolution
        else:
            return self.pre

    @property
    def post_with_physical_coordinate(self):
        """ post synapses with physical coordinate. Note that the first column is the index of
        presynapse or pre
        """
        if self.post is None:
            return None
        else:
            if self.resolution is None:
                return self.post
            else:
                post = deepcopy(self.post)
                resolution = np.asarray(self.resolution, dtype=post.dtype)
                resolution = np.expand_dims(resolution, axis=0)
                post[:, 1:] *= resolution
                return post

    @property
    def pre_index2post_indices(self):
        # pi2pi = defaultdict(list)
        pi2pi = []
        for idx in range(self.pre_num):
            # find the post synapses for this presynapse
            post_indices = np.nonzero(self.post[:, 0]==idx)
            assert len(post_indices) == 1
            post_indices = post_indices[0].tolist()
            pi2pi.append(post_indices)

        return pi2pi

    @property
    def post_synapse_num_list(self):
        pi2pi = self.pre_index2post_indices
        return [len(pi) for pi in pi2pi]

    @property
    def distances_from_pre_to_post(self):
        distances = np.zeros((self.post_num,), dtype=float)
        for post_idx in range(self.post_num):
            post = self.post[post_idx, 1:]
            pre_idx = self.post[post_idx, 0]
            pre = self.pre[pre_idx, :]
            distances[post_idx] = np.linalg.norm(pre - post)
        return distances

    @property
    def pre_indices_without_post(self) -> List[int]:
        """presynapse indices that do not have post synapses

        Returns:
            [list]: a list of presynapse indices 
        """
        pi2pi = self.pre_index2post_indices
        pre_indices = []
        for pre_index in range(self.pre_num):
            post_indices = pi2pi[pre_index]
            if len(post_indices) == 0:
                pre_indices.append(pre_index)
        return pre_indices

    def remove_pre(self, indices: List[int]):
        """remove or delete presynapses according to a list of indices
        
        Note that we need to update the post synapse as well!

        Args:
            indices (List[int]): the presynapse indices
        """# update the presynapse indices in post
        # old presynapse index to new presynapse index
        old2new = np.ones((self.pre_num,), dtype=np.int64)
        old2new[indices] = 0
        old2new = np.cumsum(old2new) - 1

        self.pre = np.delete(self.pre, indices, axis=0)
        if self.pre_confidence is not None:
            self.pre_confidence = np.delete(self.pre_confidence, indices)

        if self.post is not None: 
            post_indices = np.isin(self.post[:, 0], indices)
            self.post = np.delete(self.post, post_indices, axis=0)
            for idx in range(self.post_num):
                self.post[idx, 0] = old2new[self.post[idx, 0]]

    def remove_synapses_without_post(self):
        """remove synapse without post synapse target

        Returns:
            None: remove in place
        """
        selected_pre_indices = self.pre_indices_without_post

        # remove the selected presynapses
        self.remove_pre(selected_pre_indices)
        
    def remove_synapses_outside_bounding_box(self, bbox: BoundingBox):
        selected = []
        for idx in range(self.pre_num):
            if not bbox.contains(self.pre[idx, :]):
                selected.append(idx)

        self.remove_pre(selected)

    def remove_pre_duplicates(self):
        """some presynapses might have same coordinates.
        This is a bug in previous presynapse prediction stage.
        the bounding box contains is inclusive in both lower and upper side.
        """
        pre_index2post_indices = self.pre_index2post_indices

        raise NotImplementedError()

if __name__ == '__main__':
    synapses = Synapses.from_h5(
        os.path.expanduser(
            '~/dropbox/40_gt/21_wasp_synapses/Sp2,6848-7448_5690-6290_7038-7638.h5'
        )
    )
    assert len(synapses.pre_index2post_indices[0]) > 1
    breakpoint()