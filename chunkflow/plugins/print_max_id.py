from chunkflow.chunk import Chunk

def execute(seg: Chunk):
    print(f'maximum object ID: {seg.max()}')
