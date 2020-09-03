from typing import Union
import neuroglancer as ng
import numpy as np

from .base import OperatorBase


class NeuroglancerOperator(OperatorBase):
    def __init__(self,
                 name: str = 'neuroglancer',
                 verbose: bool = True,
                 port: int = None,
                 voxel_size: tuple = (1, 1, 1)):
        super().__init__(name=name, verbose=verbose)
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
                global_offset = chunk.global_offset
                chunk = np.ascontiguousarray(chunk)
                # neuroglancer uses F order
                chunk = np.transpose(chunk)

                # neuroglancer do not support int type
                if np.issubdtype(chunk.dtype, np.int64):
                    assert chunk.min() >= 0
                    chunk = chunk.astype(np.uint64)


                if np.ndim(chunk) == 3:
                    dimensions = ng.CoordinateSpace(
                        scales=self.voxel_size[::-1],
                        units=['nm', 'nm', 'nm'],
                        names=['x', 'y', 'z']
                    )
                elif np.ndim(chunk) == 4:
                    dimensions = ng.CoordinateSpace(
                        scales=self.voxel_size[::-1] + [1],
                        units=['nm', 'nm', 'nm', ''],
                        names=['x', 'y', 'z', 'c^']
                    )
                else:
                    raise ValueError('only support 3/4 dimension volume.')

                s.layers.append(
                    name=chunk_name,
                    layer=ng.LocalVolume(
                        data=chunk,
                        dimensions=dimensions,
                        # offset is in nm, not voxels
                        # chunkflow use C order with zyx, 
                        # while neuroglancer use F order with xyz
                        voxel_offset=global_offset[::-1][:3]
                    )
                )
        print('Open this url in browser: ')
        print(viewer)
        input('Press Enter to exit neuroglancer.')