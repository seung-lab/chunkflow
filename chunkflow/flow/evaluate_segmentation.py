from chunkflow.chunk.segmentation import Segmentation

from .base import OperatorBase


class EvaluateSegmentationOperator(OperatorBase):
    def __init__(self, name: str = 'evaluate-segmentation',
                 verbose: bool = True):
        super().__init__(name=name, verbose=verbose)

    def __call__(self, seg, groundtruth):
        seg = Segmentation(seg)
        groundtruth = Segmentation(groundtruth)

        scores = seg.evaluate(groundtruth)
        print(scores)
