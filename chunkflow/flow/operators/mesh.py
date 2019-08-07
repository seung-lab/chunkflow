import os
import json
import numpy as np
from tqdm import tqdm

from zmesh import Mesher

from cloudvolume import CloudVolume
from cloudvolume.storage import Storage

from .base import OperatorBase


class MeshOperator(OperatorBase):
    """Create mesh files from segmentation."""

    def __init__(self,
                 output_path: str,
                 output_format: str,
                 voxel_size: tuple = (40, 4, 4),
                 simplification_factor: int = 100,
                 max_simplification_error: int = 8,
                 manifest: bool = False,
                 name: str = 'meshing',
                 verbose: bool = True):
        """
        Parameters
        ------------
        output_path: 
            path to store mesh files
        output_format: 
            format of output {'ply', 'obj', 'precomputed'}
        voxel_size:
            size of voxels
        simplification_factor:
            mesh simplification factor.
        max_simplification_error:
            maximum tolerance error of meshing.
        manifest:
            create manifest files or not. This should 
            not be True if you are only doing meshing for a segmentation chunk.
        name: 
            operator name.
        verbose:
            print out informations or not.
        """
        super().__init__(name=name, verbose=verbose)
        self.simplification_factor = simplification_factor
        self.max_simplification_error = max_simplification_error
        # zmesh use fortran order, translate zyx to xyz
        self.mesher = Mesher(voxel_size[::-1])
        self.output_path = output_path
        self.output_format = output_format
        self.manifest = manifest

        if manifest:
            assert output_format == 'precomputed'

        mesh_path = output_path

        if output_format == 'precomputed':
            # adjust the mesh path according to info
            vol = CloudVolume(self.output_path)
            info = vol.info
            if 'mesh' not in info:
                # add mesh to info and update it
                info['mesh'] = 'mesh_err_{}'.format(max_simplification_error)
                vol.info = info
                vol.commit_info()
            mesh_path = os.path.join(output_path, info['mesh'])

        self.storage = Storage(mesh_path)

    def _get_mesh_data(self, obj_id):
        mesh = self.mesher.get_mesh(
            obj_id,
            normals=False,
            simplification_factor=self.simplification_factor,
            max_simplification_error=self.max_simplification_error)
        # delete high resolution mesh
        self.mesher.erase(obj_id)

        if self.output_format == 'precomputed':
            data = mesh.to_precomputed()
        elif self.output_format == 'ply':
            data = mesh.to_ply()
        elif self.output_format == 'obj':
            data = mesh.to_obj()
        else:
            raise NotImplementedError
        return data

    def __call__(self, seg: np.ndarray):
        """Meshing the segmentation.
        
        Parameters
        ------------
        seg:
            3D segmentation volume.
        """
        assert isinstance(seg, np.ndarray)
        assert seg.ndim == 3
        assert np.issubdtype(seg.dtype, np.integer)
        print('computing meshes from segmentation...')
        self.mesher.mesh(seg)

        print('write mesh to storage...')
        with self.storage as stor:
            for obj_id in tqdm(self.mesher.ids(), desc='writing out meshes'):
                data = self._get_mesh_data(obj_id)
                stor.put_file(str(obj_id), data)

                # create manifest file
                if self.manifest:
                    stor.put_file('{}:0'.format(obj_id),
                                  json.dumps({'fragments': [str(obj_id)]}))

        # release memory
        self.mesher.clear()
