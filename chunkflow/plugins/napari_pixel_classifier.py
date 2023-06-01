import numpy as np
from chunkflow.chunk import Chunk

import apoc


def execute(img: Chunk, opencl_filename: str='./PixelClassifier.cl'):
    clf = apoc.PixelClassifier(opencl_filename=opencl_filename)
    pred = clf.predict(img.array)
    pred = np.asarray(pred)
    pred = pred.astype(np.uint32)

    pred = Chunk(pred)
    pred.set_properties(img.properties)
    return pred