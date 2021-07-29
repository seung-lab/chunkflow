from typing import FrozenSet
import logging
import os
import pickle
import numpy as np


from tqdm import tqdm
from zmesh import Mesher

from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
from cloudfiles import CloudFiles

from chunkflow.chunk import Chunk
from .base import OperatorBase


class MeshOperator(OperatorBase):
    """Create mesh files from segmentation."""
    def __init__(self,
                 output_path: str,
                 output_format: str,
                 mip: int = None,
                 voxel_size: tuple = (1, 1, 1),
                 simplification_factor: int = 100,
                 max_simplification_error: int = 8,
                 manifest: bool = False,
                 skip_ids: FrozenSet[int] = None,
                 shard: bool = False,
                 name: str = 'mesh'):
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

        Note that some functions are adopted from igneous.
        """
        super().__init__(name=name)
        self.simplification_factor = simplification_factor
        self.max_simplification_error = max_simplification_error
        # zmesh use fortran order, translate zyx to xyz
        self.output_path = output_path
        self.output_format = output_format
        self.manifest = manifest
        self.skip_ids = skip_ids
        self.shard = shard

        if manifest:
            assert output_format == 'precomputed'

        if output_format == 'precomputed':
            # adjust the mesh path according to info
            vol = CloudVolume(self.output_path, mip)
            info = vol.info
            if 'mesh' not in info:
                # add mesh to info and update it
                info['mesh'] = 'mesh_err_{}'.format(max_simplification_error)
                vol.info = info
                vol.commit_info()
            self.mesh_path = os.path.join(output_path, info['mesh'])
            self.voxel_size = vol.resolution[::-1]
            self.mesher = Mesher( vol.resolution )
        else: 
            self.mesh_path = output_path
            self.mesher = Mesher(voxel_size[::-1])

        self.storage = CloudFiles(self.mesh_path)

    def _get_mesh_data(self, obj_id, offset):
        mesh = self.mesher.get_mesh(
            obj_id,
            normals=False,
            simplification_factor=self.simplification_factor,
            max_simplification_error=self.max_simplification_error)
        # delete high resolution mesh
        self.mesher.erase(obj_id)

        if self.output_format == 'precomputed':
            mesh.vertices[:] += offset[::-1] * self.voxel_size[::-1]
            data = mesh.to_precomputed()
        elif self.output_format == 'ply':
            data = mesh.to_ply()
        elif self.output_format == 'obj':
            data = mesh.to_obj()
        else:
            raise NotImplementedError

        mesh_bounds = Bbox(
            np.amin(mesh.vertices, axis=0),
            np.amax(mesh.vertices, axis=0)
        )
        return data, mesh_bounds

    def _get_file_name(self, bbox, obj_id):
        if self.output_format == 'precomputed':
            # bbox is in z,y,x order, should transform to x,y,z order 
            bbox2 = Bbox.from_slices(bbox.to_slices()[::-1])
            return '{}:0:{}'.format(obj_id, bbox2.to_filename())
        elif self.output_format == 'ply':
            return '{}.ply'.format(obj_id)
        elif self.output_format == 'obj':
            return '{}.obj'.format(obj_id)
        else:
            raise ValueError('unsupported format!')

    def __call__(self, seg: Chunk):
        """Meshing the segmentation.
        
        Parameters
        ------------
        seg:
            3D segmentation chunk.
        """
        if seg is None:
            return

        assert isinstance(seg, Chunk)
        assert seg.ndim == 3
        assert np.issubdtype(seg.dtype, np.integer)
        
        bbox = seg.bbox
        # use ndarray after getting the bounding box
        seg = seg.array

        logging.info('computing meshes from segmentation...')
        self.mesher.mesh(seg)

        logging.info('write mesh to storage...')
        if self.shard:
            assert 'precomputed' in self.output_format
            meshes = []
            mesh_bboxes = {}
            for obj_id in self.mesher.ids():
                data, mesh_bbox = self._get_mesh_data(obj_id, bbox.minpt)
                meshes.append(data)
                mesh_bboxes[obj_id] = mesh_bbox.to_list()

            # use shared format in default!
            self.storage.put(
                f"{self.mesh_path}/{bbox.to_filename()}.frags",
                content=pickle.dumps(meshes),
                compress='gzip',
                content_type="application/python-pickle",
                cache_control=False,
            )
            self.storage.put_json(
                f"{self.mesh_path}/{bbox.to_filename()}.spatial",
                mesh_bboxes,
                compress='gzip',
                cache_control=False,
            )
        else:
            if 'precomputed' in self.output_format:
                compress = 'gzip'
            else:
                compress = None

            for obj_id in tqdm(self.mesher.ids(), desc='writing out meshes'):
                if self.skip_ids is not None and obj_id in self.skip_ids:
                    continue
                # print('object id: ', obj_id)
                data, _ = self._get_mesh_data(obj_id, bbox.minpt)
                file_name = self._get_file_name(bbox, obj_id)
                self.storage.put(
                    file_name, data,
                    cache_control=None,
                    compress=compress
                )

                # create manifest file
                if self.manifest:
                    self.storage.put_json(
                        f'{obj_id}:0',
                        {'fragments': [file_name]}
                    )
                    self.storage.put_json(
                        'info',
                        {"@type": "neuroglancer_legacy_mesh"}
                    )

        # release memory
        self.mesher.clear()