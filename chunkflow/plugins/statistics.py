from chunkflow.chunk import Chunk
import fastremap


def execute(chunk: Chunk):
    if chunk.is_segmentation:
        uniq = fastremap.unique(chunk.array, return_counts=False)
        print(f'{len(uniq)} objects with min id {uniq.min()} and max id {uniq.max()}')