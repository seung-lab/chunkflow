import os
import json
import numpy as np
from tqdm import tqdm

from zmesh import Mesher
import fastremap

from cloudvolume import CloudVolume
from cloudvolume.storage import Storage
from cloudvolume.lib import Bbox

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
                 dust_threshold: int = None,
                 ids: set = None,
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
        dust_threshold:
            do not mesh tiny objects with voxel number less than threshold
        ids:
            only mesh the selected segmentation ids, other segments will not be meshed.
        manifest:
            create manifest files or not. This should 
            not be True if you are only doing meshing for a segmentation chunk.
        name: 
            operator name.
        verbose:
            print out informations or not.

        Note that some functions are adopted from igneous.
        """
        super().__init__(name=name, verbose=verbose)
        self.simplification_factor = simplification_factor
        self.max_simplification_error = max_simplification_error
        # zmesh use fortran order, translate zyx to xyz
        self.output_path = output_path
        self.output_format = output_format
        self.dust_threshold = dust_threshold
        self.ids = ids
        self.manifest = manifest

        if manifest:
            assert output_format == 'precomputed'

        mesh_path = output_path

        if output_format == 'precomputed':
            # adjust the mesh path according to info
            vol = CloudVolume(self.output_path, mip)
            info = vol.info
            if 'mesh' not in info:
                # add mesh to info and update it
                info['mesh'] = 'mesh_err_{}'.format(max_simplification_error)
                vol.info = info
                vol.commit_info()
            mesh_path = os.path.join(output_path, info['mesh'])
            self.voxel_size = vol.resolution[::-1]
            self.mesher = Mesher( vol.resolution )
        else: 
            self.mesher = Mesher(voxel_size[::-1])

        self.storage = Storage(mesh_path)

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
        return data

    def _remove_dust(self, seg: np.ndarray):
        """
        this function is adopted from igneous.
        """
        if self.verbose:
            print('remove dust segments')

        if self.dust_threshold or self.ids:
            segids, voxel_nums = np.unique(seg, return_counts=True)
            dust_segids = [sid for sid, ct in 
                            zip(segids, voxel_nums) if ct < self.dust_threshold]

            seg = fastremap.mask(seg, dust_segids, in_place=True)
        return seg

    def _only_keep_selected(self, seg: np.ndarray):
        if self.verbose:
            print('only keep selected segment ids, and remove others.')

        # precompute the remap function
        if self.ids:
            #segids = set(np.unique(seg))
            #ids = self.ids.intersection( segids )
            seg = fastremap.mask_except(seg, list(self.ids), in_place=True)
        return seg
    
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
        assert isinstance(seg, Chunk)
        assert seg.ndim == 3
        assert np.issubdtype(seg.dtype, np.integer)
        
        bbox = seg.bbox
        # use ndarray after getting the bounding box
        seg = seg.array

        seg = self._only_keep_selected(seg)
        if np.all(seg==0):
            if self.verbose:
                print('no segmentation id is selected!')
            return
        seg = self._remove_dust(seg)

        if self.verbose:
            print('computing meshes from segmentation...')
        self.mesher.mesh(seg)

        if self.verbose:
            print('write mesh to storage...')
        for obj_id in tqdm(self.mesher.ids(), desc='writing out meshes'):
            # print('object id: ', obj_id)
            data = self._get_mesh_data(obj_id, bbox.minpt)
            file_name = self._get_file_name(bbox, obj_id)
            self.storage.put_file(
                file_name, data,
                cache_control=None,
                compress='gzip'
            )

            # create manifest file
            if self.manifest:
                self.storage.put_file('{}:0'.format(obj_id),
                                json.dumps({'fragments': [str(obj_id)]}))

        # release memory
        self.mesher.clear()
