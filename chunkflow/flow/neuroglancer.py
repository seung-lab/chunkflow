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
        if 'presynapses' in data:
            presynapses = data['presynapses']
            postsynapses = data['postsynapses']
            for id, coordinate in presynapses.items():
                annotation = ng.PointAnnotation(
                    id=str(id),
                    point=coordinate[::-1],
                    # props=['#0f0', 5]
                )
                annotations.append(annotation)
                if id in postsynapses:
                    coordinates = postsynapses[id]
                    for idx, coord in enumerate(coordinates):
                        annotation = ng.PointAnnotation(
                            id=str(idx) + '_post',
                            point=coord[::-1],
                            # props=['#0ff', 5]
                        )
                        annotations.append(annotation)

        viewer_state.layers.append(
            name=name,
            layer=ng.LocalAnnotationLayer(
                dimensions=viewer_state.dimensions,
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
        voxel_offset = chunk.voxel_offset
        voxel_size = self._get_voxel_size(chunk)
        adjusted_voxel_offset = voxel_offset[::-1]
        dimensions = ng.CoordinateSpace(
            scales=voxel_size[::-1],
            units=['nm', 'nm', 'nm'],
            names=['x', 'y', 'z']
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
                # chunkflow use C order with zyx, 
                # while neuroglancer use F order with xyz
                voxel_offset=adjusted_voxel_offset,
            ),
            shader=shader
        )

    def _append_segmentation_layer(self, viewer_state: ng.viewer_state.ViewerState, chunk_name: str, chunk: Chunk):
        if np.issubdtype(chunk.dtype, np.int64):
            assert chunk.min() >= 0
            chunk = chunk.astype(np.uint64)
        voxel_offset = chunk.voxel_offset
        voxel_size = self._get_voxel_size(chunk)
        adjusted_voxel_offset = voxel_offset[::-1]
        dimensions = ng.CoordinateSpace(
            scales=voxel_size[::-1],
            units=['nm', 'nm', 'nm'],
            names=['x', 'y', 'z']
        )       
        viewer_state.layers.append(
            name=chunk_name,
            layer=ng.LocalVolume(
                data=chunk,
                dimensions=dimensions,
                # offset is in nm, not voxels
                # chunkflow use C order with zyx, 
                # while neuroglancer use F order with xyz
                voxel_offset=adjusted_voxel_offset,
            ),
            shader=None
        ) 

    def _append_affinity_map_layer(self, viewer_state: ng.viewer_state.ViewerState, chunk_name: str, chunk: Chunk):
        if chunk.dtype == np.dtype('<f4') or chunk.dtype == np.dtype('float16'):
                chunk = chunk.astype(np.float32)

        chunk = np.transpose(chunk, axes=(0, 3, 2, 1))
        voxel_offset = chunk.voxel_offset
        voxel_size = self._get_voxel_size(chunk)
        adjusted_voxel_offset = (voxel_offset[0], *voxel_offset[-3:][::-1])
        # chunk = np.transpose(chunk)
        # chunk = np.ascontiguousarray(chunk)
        dimensions = ng.CoordinateSpace(
            scales=(1, *voxel_size[::-1]),
            units=['', 'nm', 'nm', 'nm'],
            names=['c^', 'x', 'y', 'z']
        )
        shader="""void main() {
emitRGB(vec3(toNormalized(getDataValue(0)),
            toNormalized(getDataValue(1)),
            toNormalized(getDataValue(2))));
}
"""
        viewer_state.layers.append(
            name=chunk_name,
            layer=ng.LocalVolume(
                data=chunk,
                dimensions=dimensions,
                # offset is in nm, not voxels
                # chunkflow use C order with zyx, 
                # while neuroglancer use F order with xyz
                voxel_offset=adjusted_voxel_offset,
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
                    breakpoint()
                elif chunk.is_image:
                    self._append_image_layer(viewer_state, chunk_name, chunk)
                elif chunk.is_segmentation:
                    self._append_segmentation_layer(viewer_state, chunk_name, chunk)
                elif chunk.is_affinity_map:
                    self._append_affinity_map_layer(viewer_state, chunk_name, chunk)
                else:
                    raise ValueError(f'do not support this type: {type(chunk)}')

        print('Open this url in browser: ')
        print(viewer)

        key = None
        while key!='q':
            key = input('Press q and enter/return to quit neuroglancer.')
