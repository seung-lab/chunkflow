import shutil

import numpy as np

from cloudvolume.lib import generate_random_string
from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian
from chunkflow.volume import PrecomputedVolume


def test_volume():
    print('test volume cutout...')
    # compute parameters
    size = (36, 448, 448)

    # create image dataset using cloud-volume
    img = np.random.randint(0, 256, size=size)
    img = img.astype(np.uint8)
    # save the input to disk
    volume_path = 'file:///tmp/test/volume/' + \
        generate_random_string()
    
    vol = PrecomputedVolume.from_numpy(
        img,
        volume_path
    )

    offset = Cartesian(4, 64, 64)
    shape = (28, 320, 320)
    bbox = BoundingBox.from_delta(offset, shape)
    chunk = vol.cutout(bbox)
    # chunk = chunk.squeeze_channel()

    assert offset == chunk.voxel_offset
    np.testing.assert_array_equal(chunk, img[4:-4, 64:-64, 64:-64])

    shutil.rmtree('/tmp/test')