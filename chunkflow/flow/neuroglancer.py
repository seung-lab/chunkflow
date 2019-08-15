import neuroglancer as ng
import numpy as np

from .base import OperatorBase


class NeuroglancerOperator(OperatorBase):
    def __init__(self, name: str = 'neuroglancer', verbose: bool = True,
                 port: int=None, voxel_size: tuple=(1,1,1)):
        super().__init__(name=name, verbose=verbose)
        self.port = port
        self.voxel_size = voxel_size

    def __call__(self, chunks):
        """
        chunks: (list/tuple) multiple chunks 
        """
        ng.set_static_content_source(url='https://neuromancer-seung-import.appspot.com') 
        ng.set_server_bind_address(bind_port=self.port)
        viewer = ng.Viewer()

        with viewer.txn() as s:
            for idx, chunk in enumerate(chunks):
                global_offset = chunk.global_offset
                #chunk = np.transpose(chunk)
                chunk = np.ascontiguousarray(chunk)

                #chunk = np.asfortranarray(chunk)
                s.layers.append(
                    name='chunk-{}'.format(idx),
                    layer=ng.LocalVolume(
                        data=chunk,
                        voxel_size=self.voxel_size[::-1],
                        # offset is in nm, not voxels
                        offset=list(o * v for o, v in zip(
                            global_offset[::-1][-3:], self.voxel_size[::-1])),
                    ),
                    shader=get_shader(chunk),
                )
        print('Open this url in browser: ')
        print(viewer)
        input('Press Enter to exit neuroglancer.')


def get_shader(chunk):
    if chunk.ndim == 3 or chunk.shape[0]==1:
        # this is a image
        return """void main() {
    emitGrayscale(toNormalized(getDataValue()));
}"""
    elif chunk.ndim == 4 and chunk.shape[0] == 3:
        # this is affinitymap
        return """void main() {
    emitRGB(vec3(toNormalized(getDataValue(0)),
                 toNormalized(getDataValue(1)),
                 toNormalized(getDataValue(2))));
}"""
    else:
        raise ValueError('only support image and affinitymap now.')
