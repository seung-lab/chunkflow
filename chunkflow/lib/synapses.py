import os
import json

import numpy as np
import h5py


class Synapses():
    def __init__(self, tbars: np.ndarray, tbar_confidence: np.ndarray = None, 
            post_synapses: np.ndarray = None, post_synapse_confidence: np.ndarray = None, 
            resolution: tuple = None) -> None:
        """Synapses containing T-bars and post-synapses

        Args:
            tbars (np.ndarray): T-bar points, Nx3, z,y,x, the coordinate should be physical coordinate rather than voxels.
            tbar_confidence (np.ndarray, optional): confidence of T-bar detection. Defaults to None.
            post_synapses (np.ndarray, optional): [description]. Defaults to None.
            resolution (tuple, optional): [description]. Defaults to None.
        """
        assert tbars.ndim == 2
        assert tbars.shape[1] == 3

        if tbar_confidence is not None:
            assert tbar_confidence.ndim == 1
            np.testing.assert_array_less(tbar_confidence, 1.00001)
            np.testing.assert_array_less(-0.0001, tbar_confidence)
            assert len(tbar_confidence) == tbars.shape[0]

        if resolution is not None:
            resolution = np.asarray(resolution, dtype=tbars.dtype)
            np.testing.assert_array_less(0, resolution)
            tbars *= resolution
            if post_synapses is not None:
                post_synapses[:, 1:] *= post_synapses[:, 1:]

        if post_synapses is not None:
            if post_synapse_confidence is not None:
                assert post_synapse_confidence.ndim == 1
                assert len(post_synapse_confidence) == post_synapses.shape[1]

            post_synapses.ndim == 2
            # parent tbar index, z, y, x
            post_synapses.shape[1] == 4
            assert np.issubdtype(post_synapses.dtype, np.integer)

        self.tbars = tbars
        self.tbar_confidence = tbar_confidence
        self.post_synapses = post_synapses
        self.post_synapse_confidence = post_synapse_confidence
        
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

        tbar_num = len(synapses) - 2
        tbars = np.zeros((tbar_num, 3), dtype=np.int32)
        post_synapse_list = []
        tbar_indices = []
        for sid, synapse in synapses.items():
            tbars[sid, :] = np.asarray(synapse['coord'])
            if 'postsynapses' in synapse:
                for idx, post_coordinate in enumerate(synapse['postsynapses']):
                    post_synapse_list.append(post_coordinate)
                    tbar_indices.append(sid)
        
        post_synapse_num = len(post_synapse_list)
        if post_synapse_num == 0:
            post_synapses = None
        else:
            post_synapses = np.zeros((post_synapse_num, 4), dtype=np.int32)
            for idx, post_synapse in enumerate(post_synapse_list):
                post_synapses[idx, 0] = tbar_indices[idx]
                post_synapses[idx, 1:] = np.asarray(post_synapse)

        if order == ['z', 'y', 'x']:
            pass
        elif order == ['x', 'y', 'z']:
            resolution = resolution[::-1]
            # invert to z,y,x
            tbars = np.fliplr(tbars)
            if post_synapses is not None:
                post_synapses[:, 1:] = np.fliplr(post_synapses[:, 1:])

        return cls(tbars, post_synapses=post_synapses, resolution=resolution)
    
    @classmethod
    def from_json(cls, fname: str, resolution: tuple = None):
        with open(fname, 'r') as file:
            synapses = json.load(file)

        if resolution is not None:
            synapses['resolution'] = resolution
        return cls.from_dict(synapses)

    @classmethod
    def from_h5(cls, fname: str, resolution: tuple = None):
        with h5py.File(fname, 'r') as hf:
            tbars = np.asarray(hf['tbars'])
            confidence = np.asarray(hf['confidence'])
        return cls(tbars, tbar_confidence=confidence, resolution=resolution)

    @classmethod
    def from_file(cls, fname: str, resolution: tuple = None):
        assert os.path.exists(fname)
        if fname.endswith('.json'):
            return cls.from_json(fname, resolution = resolution)
        elif fname.endswith('.h5'):
            return cls.from_h5(fname, resolution=resolution)
        else:
            raise ValueError(f'only support JSON and HDF5 file, but got {fname}')
        
    @property
    def tbar_num(self):
        return self.tbars.shape[0]

