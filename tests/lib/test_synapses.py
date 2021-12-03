import os
import tempfile
import numpy as np
from chunkflow.lib.synapses import Synapses


def test_synapses():
    pre = np.arange(6).reshape((2,3))
    post = np.array([[0, 1,2,3]])
    syns = Synapses(pre, post=post)

    assert syns.pre_num == 2
    assert syns.post_num == 1

    temp_dir = tempfile.mkdtemp()
    temp_fname = os.path.join(temp_dir, 'test.h5')
    syns.to_h5(temp_fname)
    syns2 = Synapses.from_h5(temp_fname)
    assert syns == syns2

    syns.remove_synapses_without_post()
    assert syns.pre_num == 1
    