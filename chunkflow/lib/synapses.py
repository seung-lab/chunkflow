from __future__ import annotations

import os
import json
import time
from copy import deepcopy
from typing import List

import numpy as np
import h5py
from scipy.spatial import KDTree

import chunkflow
from chunkflow.chunk import Chunk
from chunkflow.lib.cartesian_coordinate import Cartesian, BoundingBox


class Synapses():
    def __init__(self, 
            pre: np.ndarray,
            pre_confidence: np.ndarray = None, 
            post: np.ndarray = None, 
            post_confidence: np.ndarray = None, 
            resolution: Cartesian = None,
            users: list = None,
            pre_users: np.ndarray = None,
            post_users: np.ndarray = None,
        ) -> None:
        """Synapses containing T-bars and post-synapses

        Args:
            pre (np.ndarray): T-bar points, Nx3, z,y,x, the coordinate should be physical coordinate rather than voxels.
            pre_confidence (np.ndarray, optional): confidence of T-bar detection. Defaults to None.
            post (np.ndarray, optional): [description]. Defaults to None.
            resolution (Cartesian, optional): [description]. Defaults to None.
            users (list[str], optional): user names of edting/ingesting the synapses.
            pre_user (np.ndarray(int8), optional): the user ids of presynapses. -1 means not identified.
            post_user (np.ndarray(int8), optional): the user ids of postsynapses. -1 means not identified.
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
                assert len(post_confidence) == post.shape[0]

            assert post.ndim == 2
            # parent pre index, z, y, x
            assert post.shape[1] == 4
            assert np.issubdtype(post.dtype, np.integer)

        if resolution is not None:
            resolution = np.asarray(resolution, dtype=pre.dtype)
            np.testing.assert_array_less(0, resolution)
        
        if users is not None:
            assert isinstance(users, list)
        if pre_users is not None:
            assert pre_users.ndim == 1 and len(pre_users) == pre.shape[0]
        if post_users is not None:
            assert post_users.ndim == 1 and len(post_users) == post.shape[0]

        self.resolution = resolution
        # unsigned integer will have minus issues
        self.pre = pre.astype(np.int32)
        self.pre_confidence = pre_confidence
        if post is not None:
            post = post.astype(np.int32)
        self.post = post
        self.post_confidence = post_confidence
        
        self.users = users
        self.pre_users = pre_users
        self.post_users = post_users

        
    @classmethod
    def from_dict(cls, syns_dict: dict):
        """Synapses as a dictionary

        Args:
            syns_dict (dict): the whole synapses in a dictionary
        """
        order = syns_dict['order']
        resolution = syns_dict['resolution']
        del syns_dict['order']
        del syns_dict['resolution']

        pre_num = len(syns_dict)
        pre = np.zeros((pre_num, 3), dtype=np.int32)
        post_list = []
        pre_indices = []
        for sid, synapse in enumerate(syns_dict.values()):
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
        print(f'loading {len(syns)} synapses...')
        pre_list = []
        post_list = []
        pre_confidence = []
        pre_users = []
        for syn in syns:
            if 'Pre' in syn['Kind']:
                # map from xyz to zyx
                pos = syn['Pos'][::-1]
                pos = Cartesian(*pos)
                pre_list.append(pos)

                if 'conf' in syn['Prop']:
                    conf = syn['Prop']['conf']
                    conf = float(conf)
                else:
                    conf = 1.0
                pre_confidence.append(conf)

                user = syn['Prop']['user']
                pre_users.append(user) 

        print('loading post synapses...')
        pre_set = set(pre_list)
        post_users = []
        for syn in syns:
            if 'Post' in syn['Kind']:
                # map from xyz to zyx
                pos = syn['Pos'][::-1]
                pos = Cartesian(*pos)
                if len(syn['Rels'])>0:
                    pre_pos = syn['Rels'][0]['To'][::-1]
                    pre_pos = Cartesian(*pre_pos)
                    if pre_pos in pre_set:
                        post_list.append((pos, pre_pos))
                        user = syn['Prop']['user']
                        post_users.append(user)
                    else:
                        print('found a postsynapse with deleted presynapse: ', syn)
                else:
                    print('found an post synapse without presynapse: ', syn)
                
                
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

        users = set(pre_users).union(set(post_users))
        users = list(users)
        user2id = {}
        for idx, user in enumerate(users):
            user2id[user] = idx
        for idx, user in enumerate(pre_users):
            pre_users[idx] = user2id[user]
        for idx, user in enumerate(post_users):
            post_users[idx] = user2id[user]

        pre_users = np.asarray(pre_users, dtype=np.int32)
        post_users = np.asarray(post_users, dtype=np.int32)
        return cls(
            pre, post=post, 
            pre_confidence=pre_confidence,
            resolution=resolution,
            users = users,
            pre_users = pre_users,
            post_users = post_users,
        )
            
    @classmethod
    def from_json(cls, fname: str, resolution: tuple = None, c_order: bool = True):
        with open(fname, 'r') as file:
            syns = json.load(file)

        if resolution is not None:
            syns['resolution'] = resolution

        syns = cls.from_dict(syns)

        if not c_order:
            syns.transpose_axis()
            
        return syns 

    @classmethod
    def from_h5(cls, fname: str, resolution: tuple = None, c_order: bool = True):
        if os.path.getsize(fname) == 0:
            print(f'synapse file is empty: {fname}')
            return None

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

            try:
                users = hf.attrs['users'].split(';')
            except:
                users = None
            #if 'users' in hf.keys():
            #    users = str(hf['users']).split(';')
            #else:
            #    users = None
            
            if 'pre_users' in hf.keys():
                pre_users = np.asarray(hf['pre_users'], dtype=np.int32)
            else:
                pre_users = None

            if 'post_users' in hf.keys():
                post_users = np.asarray(hf['post_users'], dtype=np.int32)
            else:
                post_users = None

        syns = cls(
            pre, post=post, pre_confidence=pre_confidence, 
            post_confidence=post_confidence, resolution=resolution,
            users = users,
            pre_users = pre_users,
            post_users = post_users,
        )
        if not c_order:
            # transform to C order
            syns.transpose_axis()
        return syns
    
    @classmethod
    def from_file(cls, fname: str, resolution: tuple = None, c_order: bool = True):
        assert os.path.exists(fname)
        if fname.endswith('.json'):
            return cls.from_json(fname, resolution = resolution, c_order=c_order)
        elif fname.endswith('.h5'):
            return cls.from_h5(fname, resolution=resolution, c_order=c_order)
        else:
            raise ValueError(f'only support JSON and HDF5 file, but got {fname}')

    def to_h5(self, fname: str) -> None:
        """save to a HDF5 file

        Args:
            fname (str): the file name to be saved
        """
        assert fname.endswith(".h5") or fname.endswith(".hdf5")
        print(f'save synapses to {fname}')
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

            if self.users is not None:
                hf.attrs['users'] = ';'.join(self.users)

            if self.pre_users is not None:
                hf['pre_users'] = self.pre_users
            if self.post_users is not None:
                hf['post_users'] = self.post_users

    @property
    def json_dict(self):
        """to JSON file
        """
        data = {}
        data['pre'] = self.pre.tolist()
        if self.post is not None:
            data['post'] = self.post.tolist()
        
        if self.pre_confidence is not None:
            data['pre_confidence'] = self.pre_confidence.tolist()
        
        if self.post_confidence is not None:
            data['post_confidence'] = self.post_confidence.tolist()
        
        return data

    def to_dvid_list_of_dict(self, 
            user: str = 'chunkflow',
            comment: str = 'ingested using chunkflow'):
        """convert to dictionary for bulk ingestion of synapses
        """
        data = []

        for post_idx in range(self.post_num):
            pre_idx = self.post[post_idx, 0]
            pre_coord = self.pre[pre_idx, :][::-1]
            pre_coord = [int(x) for x in pre_coord]

            post_coord = self.post_coordinates[post_idx, :][::-1]
            post_coord = [int(x) for x in post_coord]
            dic = {
                'Kind': 'PostSyn',
                'Pos': post_coord,
                'Prop': {
                    'annotation': comment,
                    'conf': str(self.post_confidence[post_idx]),
                    'user': user
                },
                'Rels': [{'Rel': 'PostSynTo', 'To': pre_coord}],
                'Tags': []
            }
            data.append(dic)

        for pre_idx, post_indices in enumerate(self.pre_index2post_indices):
            rels = []
            for post_idx in post_indices:
                post_coord = self.post_coordinates[post_idx, :][::-1]
                post_coord = [int(x) for x in post_coord]
                rels.append({
                    'Rel': 'PreSynTo', 'To': post_coord
                })

            pre_coord = self.pre[pre_idx, :][::-1]
            pre_coord = [int(x) for x in pre_coord]
            data.append({
                'Kind': 'PreSyn',
                'Pos': pre_coord,
                'Prop': {
                    'annotation': comment,
                    'conf': str(self.pre_confidence[pre_idx]),
                    'user': user
                },
                'Rels': rels,
                'Tags': []
            })
        return data
    
    def to_neutu_task(self, fname: str, 
            software_revision: int=4809,
            description: str = "transformed using chunkflow",
            file_version: int = 1,
            body_id: int = None
        ):
        """transform to a JSON file as an input to NeuTu.
        Note that current version only support presynapse. 
        There is no post-synapses transformed!

        Args:
            fname (str): file name with extension of .json
        """
        assert fname.endswith('.json')
        task = {
            'metadata': {
                "date": time.strftime('%d-%B-%Y %H:%M'),
                "session path": "",
                "software revision": software_revision,
                "description": description,
                "coordinate system": "dvid",
                "software": "chunkflow",
                "file version": file_version,
                "username": "chunkflow",
                "software version": chunkflow.version,
                "computer": "localhost"
            }
        }

        if body_id is None:
            body_id = ""

        data = []
        for idx in range(self.pre_num):
            z, y, x = self.pre[idx, :]
            data.append({
                "body ID": body_id,
                "location": [int(x), int(y), int(z)]
            })

        task['data'] = data

        with open(fname, 'w') as jf:
            json.dump(task, jf)
        return

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

    def transpose_axis(self):
        # transform to C order
        self.pre = self.pre[:, ::-1]
        self.resolution = self.resolution[::-1]
        if self.post is not None:
            self.post[:, 1:] = self.post[:, 1:][:, ::-1]
        
    def __len__(self):
        return self.post_num
    
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
            elif self.post is not None and \
                    other.post is not None and \
                    np.array_equal(self.post, other.post):
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
        if self.post is None:
            return None
        else:
            return self.post.shape[0]

    @property
    def pre_bounding_box(self) -> BoundingBox:
        bbox = BoundingBox.from_points(self.pre)
        # the end point is exclusive
        bbox = bbox.adjust((0,0,0, 1,1,1))
        return bbox

    def post_bounding_box(self) -> BoundingBox:
        bbox = BoundingBox.from_points(self.post_coordinates)
        # the right direction is exclusive
        bbox = bbox.adjust((0,0,0, 1,1,1))
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
            if self.post is None:
                post_indices = None
            else:
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
        """distance from pre to post.
        the unit is voxel rather than physical.

        Returns:
            float: the distance
        """
        distances = np.zeros((self.post_num,), dtype=np.float32)
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
        if self.pre_users is not None:
            self.pre_users = np.delete(self.pre_users, indices)

        if self.post is not None: 
            post_indices = np.isin(self.post[:, 0], indices)
            self.post = np.delete(self.post, post_indices, axis=0)
            if self.post_users is not None:
                self.post_users = np.delete(self.post_users, post_indices)
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

    def user_id(self, user: str) -> int:
        for idx, item in enumerate(self.users):
            if user == item:
                return idx 
        return None

    def post_indices_from_user(self, user: str) -> set:
        uid = self.user_id(user)
        assert uid is not None
        indices = np.nonzero(self.post_users == uid)[0]
        indices = set(indices.tolist())
        return indices

    def find_redundent_post(self, num_threshold: int = 15, distance_threshold: float = 50.) -> set:
        """remove extra number of post synapses. Only keep a maximum number of post synapses.

        Args:
            num_threshold (int): the maximum number of post synapses kept.
            distance_threshold (float): the maximum voxel distance from pre to post.

        Return:
            to_be_removed (set[int]): the post synapse indices to be removed
        """
        distances = self.distances_from_pre_to_post
        assert len(distances) == self.post_num

        # # only remove my own ingestion
        # predicted_indices = self.post_indices_from_user(user)

        to_be_removed = set()
        
        # find the distance over threshold
        indices = np.nonzero(distances > distance_threshold)
        assert len(indices) == 1
        indices = set(indices[0].tolist())
        to_be_removed = to_be_removed.union(indices)
        
        # find the extra number of post synapses
        for post_indices in self.pre_index2post_indices:
            if len(post_indices) > num_threshold:
                # we need to remove some post synapses
                dis = distances[post_indices]

                if self.post_confidence is not None:
                    # remove according to confidence
                    costs = dis / self.post_confidence[post_indices] 
                else:
                    # remove according to distance
                    costs = dis
                
                order = np.argsort(costs)
                post_indices = np.asarray(post_indices, dtype=np.int32)
                post_indices_to_remove = post_indices[order[num_threshold:]]
                to_be_removed.union(set(post_indices_to_remove.tolist()))

        # exclude the manually edited post synapses
        # to_be_removed = to_be_removed.intersection(predicted_indices)

        return to_be_removed

    def find_duplicate_post_on_same_neuron(self, seg: Chunk, distance_threshold: float=10) -> set:
        """find duplicate post synapses on the same neuron
        The T-bar could be split to two or three in a long distance

        Args:
            seg (Chunk): neuron segmentation chunk
            distance_threshold (float, optional): distance lower than this threshold is regarded as duplicate. Defaults to 10.
        
        Return:
            duplicate_indices (set[int]): a set of post synapse indices that are detected as duplicates.
        """
        post_coord = self.post_coordinates - np.asarray(seg.bbox.minpt, dtype=self.post.dtype)
        kdtree = KDTree(post_coord, leafsize=2)
        pairs = kdtree.query_pairs(distance_threshold, p=2.0, eps=0, output_type='set')

        distances = self.distances_from_pre_to_post

        duplicated_indices = set()
        def find_segid(seg: Chunk, coord: np.ndarray):
            if coord[0] >= seg.shape[0] or coord[1] >= seg.shape[1] or coord[2] >= seg.shape[2]:
                return None
            else:
                return seg[
                    coord[0],
                    coord[1],
                    coord[2]
                ]

        for idx0, idx1 in pairs:
            sid0 = find_segid(seg, post_coord[idx0, :])
            sid1 = find_segid(seg, post_coord[idx1, :])
            if sid0 is not None and sid1 is not None and sid0 == sid1 and sid0 > 0:
                if distances[idx0] > distances[idx1]:
                    duplicated_indices.add(idx0)
                else:
                    duplicated_indices.add(idx1)
        
        return duplicated_indices


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
