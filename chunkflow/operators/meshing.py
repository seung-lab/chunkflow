import numpy as np
from zmesh import Mesher
from cloudvolume.storage import Storage

from .operator_base import OperatorBase


class MeshingOperator(OperatorBase):
    def __init__(self, voxel_size=(4,4,40),
                 simplification_factor=100, 
                 max_simplification_error=8,
                 output_path='/tmp/',
                 output_format='precomputed',
                 name: str='meshing', verbose: bool=True):
        super().__init__(name=name, verbose=verbose)
        self.simplification_factor = simplification_factor
        self.max_simplification_error = max_simplification_error
        self.mesher = Mesher( voxel_size )
        self.output_path = output_path
        self.output_format = output_format

    def __call__(self, seg):
        """Meshing the segmentation."""
        assert isinstance(seg, np.ndarray)
        assert seg.ndim == 3
        self.mesher.mesh(seg)
        meshes = []
        for obj_id in self.mesher.ids():
            mesh = self.meshes.get_mesh(
                obj_id,
                normals=False,
                simplification_factor=self.simplification_factor,
                max_simplification_error=self.max_simplification_error
            )
            # delete high res mesh
            self.mesher.erase(obj_id)

            # write to storage
            mesh.to_precomputed()


        self.mesher.clear()

        # write meshes to storage
        with Storage(self.output_path) as stor: 
        for mesh in meshes:
            if self.output_format=='precomputed':
                data = mesh.to_precomputed()
            elif self.output_format=='ply':
                data = mesh.to_ply()
            elif self.output_format=='obj':
                data = mesh.to_obj()
            else:
                raise NotImplementedError
            

