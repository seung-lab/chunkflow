from typing import Union
import neuroglancer as ng
import numpy as np

from .base import OperatorBase


class NeuroglancerOperator(OperatorBase):
    def __init__(self,
                 name: str = 'neuroglancer',
                 port: int = None,
                 voxel_size: tuple = None):
        super().__init__(name=name)
        self.port = port
        self.voxel_size = voxel_size

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
        with viewer.txn() as s:
            for chunk_name in selected:
                chunk = chunks[chunk_name]
                voxel_offset = chunk.voxel_offset
                if self.voxel_size:
                    voxel_size = self.voxel_size
                elif chunk.voxel_size:
                    voxel_size = chunk.voxel_size
                else:
                    voxel_size = (1, 1, 1)

                # chunk = np.ascontiguousarray(chunk)
                # neuroglancer uses F order

                # neuroglancer do not support int type
                if np.issubdtype(chunk.dtype, np.int64):
                    assert chunk.min() >= 0
                    chunk = chunk.astype(np.uint64)
                elif chunk.dtype == np.dtype('<f4') or chunk.dtype == np.dtype('float16'):
                    chunk = chunk.astype(np.float32)

                if chunk.ndim == 3:
                    chunk = np.transpose(chunk)
                    adjusted_voxel_offset = voxel_offset[::-1]
                    dimensions = ng.CoordinateSpace(
                        scales=voxel_size[::-1],
                        units=['nm', 'nm', 'nm'],
                        names=['x', 'y', 'z']
                    )
                    if np.issubdtype(chunk.dtype, np.uint32) or \
                            np.issubdtype(chunk.dtype, np.uint64):
                        shader = None
                    else:
                        shader="""#uicontrol int channel slider(min=0, max=4)
#uicontrol vec3 color color(default="white")
#uicontrol float brightness slider(min=-1, max=1)
#uicontrol float contrast slider(min=-3, max=3, step=0.01)
void main() {
  emitRGB(color *
          (toNormalized(getDataValue(channel)) + brightness) *
          exp(contrast));
}"""
                elif chunk.ndim == 4:
                    chunk = np.transpose(chunk, axes=(0, 3, 2, 1))
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
                else:
                    raise ValueError('only support 3/4 dimension volume.')
                    
                if shader:
                    s.layers.append(
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
                else:
                    s.layers.append(
                        name=chunk_name,
                        layer=ng.LocalVolume(
                            data=chunk,
                            dimensions=dimensions,
                            # offset is in nm, not voxels
                            # chunkflow use C order with zyx, 
                            # while neuroglancer use F order with xyz
                            voxel_offset=voxel_offset[::-1],
                        ),
                    )

        print('Open this url in browser: ')
        print(viewer)

        key = None
        while key!='q':
            key = input('Press q and enter/return to quit neuroglancer.')
