from chunkflow.chunk import Chunk
import fill_voids

def execute(seg: Chunk):
    properties = seg.properties
    array = fill_voids.fill(seg.array)
    seg2 = Chunk(array)
    seg2.set_properties(properties)
    return [seg2,]