from copy import deepcopy
import numpy as np

from chunkflow.chunk import Chunk

from chunkflow.synapses import Synapses

from edt import edt


def execute(syns: Synapses, img: Chunk, img_threshold: int=60, radius: int=4) -> Synapses:
    syns2 = deepcopy(syns)

    mask = img.array < img_threshold
    dt = edt(mask)

    for idx in range(syns2.pre_num):
        coord = syns2.pre[idx, :]

        bbox = img.bbox.clone()
        shrinked_bbox = bbox.adjust(-radius)
        if not shrinked_bbox.contains(coord):
            print(f'skip this tbar: {coord}')
            continue
        coord = coord - img.voxel_offset

        box = dt[
            coord[0]-radius : coord[0]+radius,
            coord[1]-radius : coord[1]+radius,
            coord[2]-radius : coord[2]+radius,
        ]
        if box.shape != (2*radius, 2*radius, 2*radius):
            breakpoint()
        
        ind = np.unravel_index(np.argmax(box), box.shape)
        coord[0] += ind[0] - radius + img.voxel_offset[0]
        coord[1] += ind[1] - radius + img.voxel_offset[1]
        coord[2] += ind[2] - radius + img.voxel_offset[2]

        syns2.pre[idx, :] = coord

    return syns2