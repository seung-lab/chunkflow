import neuroglancer
import numpy as np

from .base import OperatorBase


class NeuroglancerOperator(OperatorBase):
    def __init__(self, name: str = 'neuroglancer', verbose: bool = True):
        super().__init__(name=name, verbose=verbose)

    def __call__(self, chunks, voxel_size=(1, 1, 1)):
        """
        chunks: (list/tuple) multiple chunks 
        """
        viewer = neuroglancer.Viewer()

        with viewer.txn() as s:
            for idx, chunk in enumerate(chunks):
                global_offset = chunk.global_offset
                #chunk = np.transpose(chunk)
                chunk = np.ascontiguousarray(chunk)
                #chunk = np.asfortranarray(chunk)
                s.layers.append(
                    name='chunk-{}'.format(idx),
                    layer=neuroglancer.LocalVolume(
                        data=chunk,
                        voxel_size=voxel_size[::-1],
                        # offset is in nm, not voxels
                        offset=list(o * v for o, v in zip(
                            global_offset[::-1][-3:], voxel_size[::-1])),
                    ),
                    shader=get_shader(chunk),
                )
        print('Open this url in browser: ')
        print(viewer)
        input('Press Enter to exit neuroglancer.')


def get_shader(chunk):
    if chunk.ndim == 3:
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
