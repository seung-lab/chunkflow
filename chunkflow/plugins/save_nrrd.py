import nrrd
from chunkflow.chunk import Chunk

def execute(chk: Chunk, file_name: str=None):
    if file_name is None:
        file_name = f'{chk.bbox.string}.nrrd'
    elif not file_name.endswith('.nrrd'):
        file_name += f'_{chk.bbox.string}.nrrd'

    print(f'write chunk to file: {file_name}')
    nrrd.write(file_name, chk.array)

