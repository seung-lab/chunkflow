
from skimage.exposure import rescale_intensity

from chunkflow.chunk import Chunk

def execute(img: Chunk, low: int = None, high: int = None):
    img.array = rescale_intensity(img.array, in_range=(low, high), out_range=(0, 255))
    return img