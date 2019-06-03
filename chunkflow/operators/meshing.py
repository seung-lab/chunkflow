import json
import numpy as np
from zmesh import Mesher
from cloudvolume.storage import Storage

from .operator_base import OperatorBase


class MeshingOperator(OperatorBase):
    def __init__(self, voxel_size=(4, 4, 40),
                 simplification_factor: int = 100,
                 max_simplification_error: int = 8,
                 output_path: str = '/tmp/',
                 output_format: str = 'precomputed',
                 create_manifest: bool = True,
                 name: str = 'meshing', verbose: bool = True):
        super().__init__(name=name, verbose=verbose)
        self.simplification_factor = simplification_factor
        self.max_simplification_error = max_simplification_error
        self.mesher = Mesher(voxel_size)
        self.output_path = output_path
        self.output_format = output_format
        self.create_manifest = create_manifest

    def __call__(self, seg):
        """Meshing the segmentation."""
        assert isinstance(seg, np.ndarray)
        assert seg.ndim == 3
        self.mesher.mesh(seg)

        for obj_id in self.mesher.ids():
            mesh = self.mesher.get_mesh(
                obj_id,
                normals=False,
                simplification_factor=self.simplification_factor,
                max_simplification_error=self.max_simplification_error
            )
            # delete high resolution mesh
            self.mesher.erase(obj_id)

            # write to storage
            mesh.to_precomputed()
            # write meshes to storage
            with Storage(self.output_path) as stor:
                if self.output_format == 'precomputed':
                    data = mesh.to_precomputed()
                elif self.output_format == 'ply':
                    data = mesh.to_ply()
                elif self.output_format == 'obj':
                    data = mesh.to_obj()
                else:
                    raise NotImplementedError
                stor.put_file(str(obj_id), data)

                # create manifest file
                if self.create_manifest:
                    stor.put_file('{}:0'.format(obj_id),
                                  json.dumps({'fragments': [str(obj_id)]}))

        # release memory        
        self.mesher.clear()
