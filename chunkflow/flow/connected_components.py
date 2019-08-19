from chunkflow.chunk import Chunk
import cc3d

from .base import OperatorBase


class ConnectedComponentsOperator(OperatorBase):
    """Threshold the map chunk and get the connected components label."""
    def __init__(self, name: str = 'view', verbose: bool = True,
                 threshold: float = 0.5, connectivity: int = 26):
        super().__init__(name=name, verbose=verbose)
        self.threshold = threshold
        self.connectivity = connectivity

    def __call__(self, chunk):
        """threshold the map chunk and get connected components."""
        if chunk.ndim == 4 and chunk.shape[0] == 1:
            # cc3d only support 2D/3D array, squeeze the first channel
            chunk = chunk.squeeze_channel()

        global_offset = chunk.global_offset
        seg = chunk > self.threshold
        seg = cc3d.connected_components(seg, connectivity=self.connectivity)
        return Chunk(seg, global_offset=global_offset)
