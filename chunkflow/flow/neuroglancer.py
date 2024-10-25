"""
TO-DO:
add probability map layer for T-bar or cleft detection and other semantic prediction
"""
from typing import Tuple, DefaultDict
from collections import defaultdict

import neuroglancer as ng
import numpy as np
from tqdm import tqdm

from chunkflow.chunk import Chunk
from chunkflow.synapses import Synapses
from chunkflow.point_cloud import PointCloud
from cloudvolume import Skeleton

from .base import OperatorBase


class SkeletonSource(ng.skeleton.SkeletonSource):
    def __init__(self, oid2skel: DefaultDict[int, Skeleton], scales: Tuple = (1, 1, 1), voxel_offset=None):

        dimensions = ng.CoordinateSpace(
            names=["x", "y", "z"],
            units="nm",
            scales=scales,
        )
        super().__init__(dimensions, voxel_offset)
        self.oid2skel = oid2skel
    
    def get_skeleton(self, oid: int):
        skel = self.oid2skel[oid]
        skel = ng.skeleton.Skeleton(skel.vertices, skel.edges)
        return skel


class NeuroglancerOperator(OperatorBase):
    def __init__(self,
                 name: str = 'neuroglancer',
                 port: int = None,
                 voxel_size: Tuple = None):
        super().__init__(name=name)
        self.port = port
        self.voxel_size = voxel_size
    
    def _get_voxel_size(self, chunk):
        if self.voxel_size:
            voxel_size = self.voxel_size
        elif chunk.voxel_size:
            voxel_size = chunk.voxel_size
        else:
            voxel_size = (1, 1, 1)
        
        return voxel_size

    def _append_skeleton_layer(self, 
            viewer_state: ng.viewer_state.ViewerState, 
            name: str, 
            oid2skel: DefaultDict):

        annotations = []
        for oid, skel in tqdm(oid2skel.items(), 
                              desc='make skeleton line segments'):
            # switch the X and Y to align with the image!
            # I still do not understand why should we do this!
            skel.vertices[:, 0], skel.vertices[:, 1] = skel.vertices[:, 1], skel.vertices[:, 0] 
            for p1, p2 in skel.edges:
                # breakpoint()
                ann = ng.viewer_state.LineAnnotation(
                    id = oid,
                    point_a = skel.vertices[p1, :],
                    point_b = skel.vertices[p2, :],
                )
                annotations.append(ann)
        print(f'number of annotations: {len(annotations)}')

        viewer_state.layers.append(
            name = name,
            layer = ng.viewer_state.LocalAnnotationLayer(
                dimensions=ng.CoordinateSpace(
                    names=['x', 'y', 'z'],
                    units='nm',
                    scales=(1,1,1),
                ),
                annotation_properties=[
                    ng.viewer_state.AnnotationPropertySpec(
                        id='color',
                        type='rgb',
                        default='red',
                    ),
                    ng.viewer_state.AnnotationPropertySpec(
                        id='size',
                        type='float32',
                        default=2,
                    ),
                ],
                annotations=annotations,
                shader='''
void main() {
setColor(prop_color());
setPointMarkerSize(prop_size());
}
'''
            )
        )

    def _append_synapse_annotation_layer(self, 
            viewer_state: ng.viewer_state.ViewerState, 
            name: str, 
            synapses: Synapses):
        annotations = []
        
        pre_synapses = synapses.pre_with_physical_coordinate
        post_synapses = synapses.post_with_physical_coordinate
        
        if post_synapses is not None:
            for post_idx in range(post_synapses.shape[0]):
                pre_idx = post_synapses[post_idx, 0]
                pre_coordinate = pre_synapses[pre_idx, :]
                post_coordinate = post_synapses[post_idx, 1:]
                post_annotation = ng.LineAnnotation(
                        id=str(post_idx),
                        pointA=pre_coordinate[::-1],
                        pointB=post_coordinate[::-1],
                        props=['#0ff', 5]
                    )
                annotations.append(post_annotation) 

        viewer_state.layers.append(
            name=name,
            layer=ng.LocalAnnotationLayer(
                dimensions=ng.CoordinateSpace(
                    names=['x', 'y', 'z'], 
                    units="nm", 
                    scales=(1,1,1)
                ),
                annotation_properties=[
                    ng.AnnotationPropertySpec(
                        id='color',
                        type='rgb',
                        default='red',
                    ),
                    ng.AnnotationPropertySpec(
                        id='size',
                        type='float32',
                        default=5
                    )
                ],
                annotations=annotations,
                shader='''
void main() {
  setColor(prop_color());
  setPointMarkerSize(prop_size());
}
''',
            ),
        )

        self._append_point_annotation_layer(
            viewer_state, name + '_pre', synapses.pre_point_cloud)

    def _append_point_annotation_layer(self, 
            viewer_state: ng.viewer_state.ViewerState, 
            name: str, points: PointCloud, 
            color: str = '#ff0', size: int = 8):
        annotations = []

        for sid in range(points.point_num):
            # we would like to show line first and then the presynapse point
            # so, we have distinct color to show T-bar
            pre_annotation = ng.PointAnnotation(
                id=str(sid),
                point=points.points[sid, :].tolist()[::-1],
                props=[color, size]
            )
            annotations.append(pre_annotation)

        viewer_state.layers.append(
            name=name,
            layer=ng.LocalAnnotationLayer(
                dimensions=ng.CoordinateSpace(
                    names=['x', 'y', 'z'], 
                    units="nm", 
                    scales=(
                        points.voxel_size.x, 
                        points.voxel_size.y, 
                        points.voxel_size.z, 
                    )
                ),
                annotation_properties=[
                    ng.AnnotationPropertySpec(
                        id='color',
                        type='rgb',
                        default='red',
                    ),
                    ng.AnnotationPropertySpec(
                        id='size',
                        type='float32',
                        default=5
                    )
                ],
                annotations=annotations,
                shader='''
void main() {
  setColor(prop_color());
  setPointMarkerSize(prop_size());
}
''',
            ),
        )

    def _append_image_layer(self, viewer_state: ng.viewer_state.ViewerState, chunk_name: str, chunk: Chunk):
        if chunk.ndim == 3:
            arr = chunk.array.transpose()
            names=['x', 'y', 'z']
            units=['nm', 'nm', 'nm']
            voxel_offset = chunk.voxel_offset[::-1]
            voxel_size = chunk.voxel_size[::-1]
            shader="""#uicontrol invlerp normalized
void main() {
  emitGrayscale(normalized());
}"""
        elif chunk.ndim == 4:
            arr = chunk.array.transpose()
            names=['x', 'y', 'z', 'c']
            units=['nm', 'nm', 'nm', '']
            voxel_offset = (
                chunk.voxel_offset.x, 
                chunk.voxel_offset.y, 
                chunk.voxel_offset.z,
                0,
            )
            voxel_size = (*chunk.voxel_size[::-1], 1)
            shader="""#uicontrol int channel slider(min=0, max=4)
#uicontrol vec3 color color(default="white")
#uicontrol float brightness slider(min=-1, max=1)
#uicontrol float contrast slider(min=-3, max=3, step=0.01)
void main() {
  emitRGB(color *
          (toNormalized(getDataValue(channel)) + brightness) *
          exp(contrast));
}"""
        else:
            raise ValueError(f'we only support 3/4 D, but got {chunk.ndim}')

        dimensions = ng.CoordinateSpace(
            scales=voxel_size,
            units=units,
            names=names
        )
        viewer_state.layers.append(
            name=chunk_name,
            layer=ng.LocalVolume(
                data=arr,
                dimensions=dimensions,
                voxel_offset=voxel_offset,
            ),
            shader=shader
        )

    def _append_segmentation_layer(self, viewer_state: ng.viewer_state.ViewerState, chunk_name: str, chunk: Chunk):

        if np.issubdtype(chunk.dtype, bool):
            # neuroglancer do not recognize bool type as segmentation
            chunk = chunk.astype(np.uint8)

        if np.issubdtype(chunk.dtype, np.int64):
            assert chunk.min() >= 0
            chunk = chunk.astype(np.uint64)
        elif np.issubdtype(chunk.dtype, np.uint8):
            chunk = chunk.astype(np.uint32)
        voxel_size = self._get_voxel_size(chunk)
        dimensions = ng.CoordinateSpace(
            scales=voxel_size[::-1],
            units=['nm', 'nm', 'nm'],
            names=['x', 'y', 'z']
        )       
        viewer_state.layers.append(
            name=chunk_name,
            layer=ng.LocalVolume(
                data=chunk.transpose(),
                dimensions=dimensions,
                voxel_offset=chunk.voxel_offset[::-1],
            )
        ) 

    def _append_probability_map_layer(self,
            viewer_state: ng.viewer_state.ViewerState,
            chunk_name: str, chunk: Chunk, color=None):
        if chunk.dtype == np.dtype('<f4') or chunk.dtype == np.dtype('float16'):
            chunk = chunk.astype(np.float32)

        voxel_size = self._get_voxel_size(chunk)
        # chunk = np.ascontiguousarray(chunk)
        if chunk.shape[0] == 1:
            if color is not None:
                shader = """#uicontrol vec3 color color(default="%s")
#uicontrol float brightness slider(min=-1, max=1)
#uicontrol float contrast slider(min=-3, max=3, step=0.01)
void main() {
  emitRGB(color *
          (toNormalized(getDataValue(0)) + brightness) * exp(contrast));
}
""" % color
            else:
                shader = """void main() {
emitGrayscale(toNormalized(getDataValue(0)));
}
""" 
        elif chunk.shape[0] == 2:
            shader = """void main() {
emitRGB(vec3(toNormalized(getDataValue(0)),
            toNormalized(getDataValue(1)),
            0.));
}
"""
        else:
            shader = """void main() {
emitRGB(vec3(toNormalized(getDataValue(0)),
            toNormalized(getDataValue(1)),
            toNormalized(getDataValue(2))));
}
"""
        dimensions = ng.CoordinateSpace(
            scales= voxel_size[::-1] + (1, ) ,
            units=['nm', 'nm', 'nm', ''],
            names=[ 'x', 'y', 'z', 'c^']
        )
        viewer_state.layers.append(
            name=chunk_name,
            layer=ng.LocalVolume(
                data=chunk.array.transpose(),
                dimensions=dimensions,
                # offset is in nm, not voxels
                voxel_offset=(*chunk.voxel_offset[::-1], 0),
            ),
            shader=shader
        )

    def __call__(self, datas: dict, selected: str=None):
        """
        Parameters:
        chunks: multiple chunks
        """
        def parse_selected_args(varname: str) -> Tuple[str, dict]:
            kws = {}
            if '[' in varname:
                if not varname.endswith(']'):
                    raise ValueError(f"Unmatched bracket in variable name: '{varname}'")
                varname, opts = varname[:-1].split('[')
                for arg in opts.split(','):
                    if '=' in arg:
                        k, v = arg.split('=')
                        kws[k] = v
                    else:
                        raise ValueError("Only keyword arguments are allowed in neuroglancer variable options")
            elif ']' in varname:
                raise ValueError(f"Unmatched bracket in variable name: '{varname}'")

            return varname, kws

        if selected is None:
            selected = datas.keys()
        elif isinstance(selected, str):
            selected = selected.split(',')

        # ng.set_static_content_source(
        #     url='https://neuromancer-seung-import.appspot.com')
        ng.set_server_bind_address(bind_address='0.0.0.0', bind_port=self.port)
        viewer = ng.Viewer()
        with viewer.txn() as viewer_state:
            for name in selected:
                name, layer_kwargs = parse_selected_args(name)
                data = datas[name]
                layer_args = (viewer_state, name, data)
                # breakpoint()
                
                if data is None:
                    continue
                elif isinstance(data, PointCloud):
                    # points
                    self._append_point_annotation_layer(*layer_args, **layer_kwargs)
                elif isinstance(data, Synapses):
                    # this could be synapses
                    self._append_synapse_annotation_layer(*layer_args, **layer_kwargs)
                elif (isinstance(data, defaultdict) or isinstance(data, dict)) \
                        and len(data)>0:
                    self._append_skeleton_layer(*layer_args, **layer_kwargs)
                elif isinstance(data, np.ndarray) and 2 == data.ndim and 3 == data.shape[1]:
                    # points
                    self._append_point_annotation_layer(*layer_args, **layer_kwargs)
                elif isinstance(data, Chunk):
                    if data.layer_type is None:
                        if data.is_image:
                            self._append_image_layer(*layer_args, **layer_kwargs)
                        elif data.is_segmentation:
                            self._append_segmentation_layer(*layer_args, **layer_kwargs)
                        elif data.is_probability_map:
                            self._append_probability_map_layer(*layer_args, **layer_kwargs)
                        elif data.is_affinity_map:
                            raise ValueError('affinity map is not working yet. To-Do.')
                        else:
                            raise ValueError('unsupported data type.')
                    if data.layer_type == 'segmentation':
                        self._append_segmentation_layer(*layer_args, **layer_kwargs)
                    elif data.layer_type == 'probability_map':
                        self._append_probability_map_layer(*layer_args, **layer_kwargs)
                    elif data.layer_type in set(['image', 'affinity_map']):
                        self._append_image_layer(*layer_args, **layer_kwargs)
                    else:
                        breakpoint()
                        raise ValueError('only support image, affinity map, probability_map, and segmentation for now.')
                else:
                    breakpoint()
                    raise ValueError(f'do not support this type: {type(data)}')

        print('Open this url in browser: ')
        viewer_url = viewer.get_viewer_url()
        print(viewer_url)

        key = None
        while key!='q':
            key = input('Press q and enter/return to quit neuroglancer.')
