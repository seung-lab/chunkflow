import numpy as np

from chunkflow.chunk import Chunk
from chunkflow.lib.synapses import Synapses
from chunkflow.lib.cartesian_coordinate import BoundingBox

OUT_DIR = "/mnt/ceph/users/neuro/wasp_em/jwu/sample1/75_duplicated_post"

def execute(bbox: BoundingBox, syns: Synapses, seg: Chunk, args: str = None):
    if syns is None:
        return

    MAX_POST_NUM = int(args)

    to_be_removed = syns.find_redundent_post(num_threshold=MAX_POST_NUM, distance_threshold=50.) 

    # find the post synapses on the same neuron within a distance
    same_neuron_set = syns.find_duplicate_post_on_same_neuron(seg)
    to_be_removed = to_be_removed.union(same_neuron_set) 

    # keep the proofread ones
    predicted = syns.post_indices_from_user('jwu')
    to_be_removed = to_be_removed.intersection(predicted)

    # transform to array of points
    to_be_removed = list(to_be_removed)
    posts = syns.post[to_be_removed, :]
    pres = syns.pre[posts[:, 0], :]
    # save the result
    # to_be_removed = np.asarray(to_be_removed, dtype=np.int32)
    fname = f'{OUT_DIR}/zyx_{bbox.string}.npz'
    posts = posts[:, 1:]
    np.savez(fname, pres, posts)

