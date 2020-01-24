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

    def __call__(self, chunks: dict):
        """
        Parameters:
        chunks: multiple chunks 
        """
        ng.set_static_content_source(
            url='https://neuromancer-seung-import.appspot.com')
        ng.set_server_bind_address(bind_port=self.port)
        viewer = ng.Viewer()

        with viewer.txn() as s:
            for chunk_name, chunk in chunks.items():
                global_offset = chunk.global_offset
                chunk = np.ascontiguousarray(chunk)

                s.layers.append(
                    name=chunk_name,
                    layer=ng.LocalVolume(
                        data=chunk,
                        dimensions=neuroglancer.CordinateSpace(
                            scales=[1, *self.voxel_size[::-1]],
                            units = ['', 'nm', 'nm', 'nm'],
                            names = ['c^', 'x', 'y', 'z']
                        ),
                        # offset is in nm, not voxels
                        offset=list(o * v for o, v in zip(
                            global_offset[::-1][-3:], self.voxel_size[::-1])))
                )
        print('Open this url in browser: ')
        print(viewer)
        input('Press Enter to exit neuroglancer.')
