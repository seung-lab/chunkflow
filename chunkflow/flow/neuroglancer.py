"""
TO-DO:
add probability map layer for T-bar or cleft detection and other semantic prediction
"""

import neuroglancer as ng
import numpy as np

from chunkflow.chunk import Chunk
from chunkflow.lib.synapses import Synapses

from .base import OperatorBase


class NeuroglancerOperator(OperatorBase):
    def __init__(self,
                 name: str = 'neuroglancer',
                 port: int = None,
                 voxel_size: tuple = None):
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

    def _append_synapse_annotation_layer(self, viewer_state: ng.viewer_state.ViewerState, name: str, synapses: Synapses):
        annotations = []
        
        pre_synapses = synapses.pre_with_physical_coordinate
        self._append_point_annotation_layer(viewer_state, name + '_pre', pre_synapses)

        post_synapses = synapses.post_with_physical_coordinate
        if post_synapses is not None:
            for post_idx in range(post_synapses.shape[0]):
                pre_idx = post_synapses[post_idx, 0]
                pre_coordinate = pre_synapses[pre_idx, :]
                post_coordinate = post_synapses[post_idx, 1:]
                post_annotation = ng.LineAnnotation(
                        id=str(post_idx),
                        # note that the synapse coordinate is already in xyz order
                        # so we do not need to reverse it!
                        pointA=pre_coordinate,
                        pointB=post_coordinate,
                        props=['#0ff', 5]
                    )
                annotations.append(post_annotation) 

        viewer_state.layers.append(
            name=name,
            layer=ng.LocalAnnotationLayer(
                dimensions=ng.CoordinateSpace(names=['z', 'y', 'x'], units="nm", scales=(1,1,1)),
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

    def _append_point_annotation_layer(self, viewer_state: ng.viewer_state.ViewerState, name: str, points: np.ndarray):
        annotations = []
        
        for sid in range(points.shape[0]):
            # we would like to show line first and then the presynapse point
            # so, we have distinct color to show T-bar
            pre_annotation = ng.PointAnnotation(
                id=str(sid),
                point=points[sid, :].tolist(),
                props=['#ff0', 8]
            )
            annotations.append(pre_annotation)

        viewer_state.layers.append(
            name=name,
            layer=ng.LocalAnnotationLayer(
                dimensions=ng.CoordinateSpace(names=['z', 'y', 'x'], units="nm", scales=(1, 1, 1)),
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
        voxel_size = self._get_voxel_size(chunk)
        dimensions = ng.CoordinateSpace(
            scales=voxel_size,
            units=['nm', 'nm', 'nm'],
            names=['z', 'y', 'x']
        )
        shader="""#uicontrol int channel slider(min=0, max=4)
#uicontrol vec3 color color(default="white")
#uicontrol float brightness slider(min=-1, max=1)
#uicontrol float contrast slider(min=-3, max=3, step=0.01)
void main() {
  emitRGB(color *
          (toNormalized(getDataValue(channel)) + brightness) *
          exp(contrast));
}"""
        viewer_state.layers.append(
            name=chunk_name,
            layer=ng.LocalVolume(
                data=chunk,
                dimensions=dimensions,
                # offset is in nm, not voxels
                voxel_offset=chunk.voxel_offset,
            ),
            shader=shader
        )

    def _append_segmentation_layer(self, viewer_state: ng.viewer_state.ViewerState, chunk_name: str, chunk: Chunk):
        if np.issubdtype(chunk.dtype, np.int64):
            assert chunk.min() >= 0
            chunk = chunk.astype(np.uint64)
        voxel_size = self._get_voxel_size(chunk)
        dimensions = ng.CoordinateSpace(
            scales=voxel_size,
            units=['nm', 'nm', 'nm'],
            names=['z', 'y', 'x']
        )       
        viewer_state.layers.append(
            name=chunk_name,
            layer=ng.LocalVolume(
                data=chunk,
                dimensions=dimensions,
                # offset is in nm, not voxels
                # chunkflow use C order with zyx, 
                # while neuroglancer use F order with xyz
                voxel_offset=chunk.voxel_offset,
            )
        ) 

    def _append_probability_map_layer(self, viewer_state: ng.viewer_state.ViewerState, chunk_name: str, chunk: Chunk):
        if chunk.dtype == np.dtype('<f4') or chunk.dtype == np.dtype('float16'):
            chunk = chunk.astype(np.float32)

        voxel_size = self._get_voxel_size(chunk)
        # chunk = np.ascontiguousarray(chunk)
        if chunk.shape[0] == 1:
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
            scales=(1, ) + voxel_size,
            units=['', 'nm', 'nm', 'nm'],
            names=['c^', 'z', 'y', 'x']
        )
        viewer_state.layers.append(
            name=chunk_name,
            layer=ng.LocalVolume(
                data=chunk.array,
                dimensions=dimensions,
                # offset is in nm, not voxels
                voxel_offset=(0, ) + chunk.voxel_offset,
            ),
            shader=shader
        )

    def __call__(self, datas: dict, selected: str=None):
        """
        Parameters:
        chunks: multiple chunks
        """
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
                data = datas[name]
                if data is None:
                    continue
                elif isinstance(data, Synapses):
                    # this could be synapses
                    self._append_synapse_annotation_layer(viewer_state, name, data)
                elif isinstance(data, np.ndarray) and 2 == data.ndim and 3 == data.shape[1]:
                    # points
                    self._append_point_annotation_layer(viewer_state, name, data)
                elif data.is_image or (data.ndim==3 and np.issubdtype(data.dtype, np.floating)):
                    self._append_image_layer(viewer_state, name, data)
                elif data.is_segmentation:
                    self._append_segmentation_layer(viewer_state, name, data)
                elif data.is_probability_map:
                    self._append_probability_map_layer(viewer_state, name, data)
                else:
                    breakpoint()
                    raise ValueError(f'do not support this type: {type(data)}')

        print('Open this url in browser: ')
        viewer_url = viewer.get_viewer_url()
        print(viewer_url)

        key = None
        while key!='q':
            key = input('Press q and enter/return to quit neuroglancer.')
