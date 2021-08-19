"""
TO-DO:
add probability map layer for T-bar or cleft detection and other semantic prediction
"""

import neuroglancer as ng
import numpy as np

from chunkflow.chunk import Chunk

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

    def _append_synapse_annotation_layer(self, viewer_state: ng.viewer_state.ViewerState, name: str, data: dict):
        annotations = []
        
        for sid, synapse in data.items():
            if sid in ['order', 'resolution']:
                continue

            pre_coordinate = synapse['coord']

            # breakpoint()
            if 'postsynapses' in synapse:
                postsynapses = synapse['postsynapses']
                # print(f'found {len(postsynapses)} post-synapses.')
                for idx, post_coordinate in enumerate(postsynapses):
                    post_annotation = ng.LineAnnotation(
                        id=str(sid) + str(idx) + '_post',
                        # note that the synapse coordinate is already in xyz order
                        # so we do not need to reverse it!
                        pointA=pre_coordinate,
                        pointB=post_coordinate,
                        props=['#0ff', 5]
                    )
                    annotations.append(post_annotation)
            # we would like to show line first and then the presynapse point
            # so, we have distinct color to show T-bar
            pre_annotation = ng.PointAnnotation(
                id=str(sid) + '_pre',
                point=pre_coordinate,
                props=['#ff0', 8]
            )
            annotations.append(pre_annotation)

        viewer_state.layers.append(
            name=name,
            layer=ng.LocalAnnotationLayer(
                dimensions=ng.CoordinateSpace(names=data['order'], units="nm", scales=data['resolution']),
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

    def __call__(self, chunks: dict, selected: str=None):
        """
        Parameters:
        chunks: multiple chunks
        """
        if selected is None:
            selected = chunks.keys()
        elif isinstance(selected, str):
            selected = selected.split(',')

        # ng.set_static_content_source(
        #     url='https://neuromancer-seung-import.appspot.com')
        ng.set_server_bind_address(bind_address='0.0.0.0', bind_port=self.port)
        viewer = ng.Viewer()
        with viewer.txn() as viewer_state:
            for chunk_name in selected:
                chunk = chunks[chunk_name]
                if isinstance(chunk, dict):
                    # this could be synapses
                    self._append_synapse_annotation_layer(viewer_state, chunk_name, chunk)
                elif chunk.is_image or (chunk.ndim==3 and np.issubdtype(chunk.dtype, np.floating)):
                    self._append_image_layer(viewer_state, chunk_name, chunk)
                elif chunk.is_segmentation:
                    self._append_segmentation_layer(viewer_state, chunk_name, chunk)
                elif chunk.is_probability_map:
                    self._append_probability_map_layer(viewer_state, chunk_name, chunk)
                else:
                    breakpoint()
                    raise ValueError(f'do not support this type: {type(chunk)} with datatype {chunk.dtype}')

        print('Open this url in browser: ')
        viewer_url = viewer.get_viewer_url()
        print(viewer_url)

        key = None
        while key!='q':
            key = input('Press q and enter/return to quit neuroglancer.')
