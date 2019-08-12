from chunkflow.chunk.segmentation import Segmentation

from .base import OperatorBase


class CreateChunkOperator(OperatorBase):
    def __init__(self, name: str = 'evaluate-segmentation',
                 verbose: bool = True):
        super().__init__(name=name, verbose=verbose)

    def __call__(self, seg1, seg2):
        seg1 = Segmentation(seg1)
        seg2 = Segmentation(seg2)

        scores = seg1.evaluate(seg2)
        print(scores)
