import os
import json
from copy import deepcopy
from functools import cached_property
from collections import defaultdict

import numpy as np
import h5py

from .bounding_boxes import BoundingBox


class Synapses():
    def __init__(self, pre: np.ndarray, pre_confidence: np.ndarray = None, 
            post: np.ndarray = None, post_confidence: np.ndarray = None, 
            resolution: tuple = None) -> None:
        """Synapses containing T-bars and post-synapses

        Args:
            pre (np.ndarray): T-bar points, Nx3, z,y,x, the coordinate should be physical coordinate rather than voxels.
            pre_confidence (np.ndarray, optional): confidence of T-bar detection. Defaults to None.
            post (np.ndarray, optional): [description]. Defaults to None.
            resolution (tuple, optional): [description]. Defaults to None.
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
        self.post = post.astype(np.int32)
        self.post_confidence = post_confidence
        
    @classmethod
    def from_dict(cls, synapses: dict):
        """Synapses as a dictionary

        Args:
            synapses (dict): the whole synapses in a dictionary
        """
        order = synapses['order']
        resolution = synapses['resolution']
        del synapses['order']
        del synapses['resolution']

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
    def from_json(cls, fname: str, resolution: tuple = None):
        with open(fname, 'r') as file:
            synapses = json.load(file)

        if resolution is not None:
            synapses['resolution'] = resolution
        return cls.from_dict(synapses)

    @classmethod
    def from_h5(cls, fname: str, resolution: tuple = None, c_order: bool = True):
        with h5py.File(fname, 'r') as hf:

            pre = np.asarray(hf['pre'], dtype=np.int32)

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
        
    @property
    def pre_num(self):
        return self.pre.shape[0]
    
    @property
    def post_num(self):
        return self.post.shape[0]

    @property
    def pre_with_physical_coordinate(self):
        if self.resolution is not None:
            return self.pre * self.resolution
        else:
            return self.pre
        
    @property
    def bounding_box(self) -> BoundingBox:
        bbox = BoundingBox.from_points(self.pre)
        bbox_post = BoundingBox.from_points(self.post[:, 1:])
        bbox.union(bbox_post)
        return bbox
                    
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
                post[:, 1:] *= self.resolution
                return post

    @cached_property
    def pre_index2post_indices(self):
        pi2pi = defaultdict(list)
        for idx in range(self.pre_num):
            # find the post synapses for this presynapse
            post_indices = np.argwhere(self.post[:, 0]==idx)
            pi2pi[idx].append(post_indices)

        return pi2pi

    @cached_property
    def distances_from_pre_to_post(self):
        distances = np.zeros((self.post_num,), dtype=float)
        for post_idx in range(self.post_num):
            post = self.post[post_idx, 1:]
            pre_idx = self.post[post_idx, 0]
            pre = self.pre[pre_idx, :]
            distances[post_idx] = np.linalg.norm(pre - post)
        return distances
    
