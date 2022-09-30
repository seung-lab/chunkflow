import cv2

from chunkflow.chunk import Chunk

clahe = cv2.createCLAHE(clipLimit=5., tileGridSize = (18, 26))

def execute(img: Chunk):
    assert img.is_image
    ret = clahe.apply(img.array)
    ret = Chunk(ret)
    ret.set_properties(img.properties)
    return [ret]
