import unittest
import shutil
import numpy as np

from cloudvolume import CloudVolume
from cloudvolume.storage import Storage
from cloudvolume.lib import generate_random_string, Bbox

from chunkflow.flow.cutout import CutoutOperator


class TestCutout(unittest.TestCase):
    def setUp(self):
        print('test volume cutout...')
        # compute parameters
        self.mip = 0
        self.size = (36, 448, 448)

        # create image dataset using cloud-volume
        img = np.random.randint(0, 256, size=self.size)
        self.img = img.astype(np.uint8)
        # save the input to disk
        self.volume_path = 'file:///tmp/test/cutout/' + generate_random_string(
        )
        CloudVolume.from_numpy(np.transpose(self.img),
                               vol_path=self.volume_path)

        # prepare blackout section ids
        self.blackout_section_ids = [17, 20]
        ids = {'section_ids': self.blackout_section_ids}
        with Storage(self.volume_path) as stor:
            stor.put_json('blackout_section_ids.json', ids)

    def test_cutout(self):
        print('test volume cutout...')
        operator = CutoutOperator(self.volume_path, mip=self.mip)

        offset = (4, 64, 64)
        shape = (28, 320, 320)
        output_bbox = Bbox.from_delta(offset, shape)
        chunk = operator(output_bbox)

        self.assertEqual(offset, chunk.global_offset)
        self.assertTrue(chunk == self.img[4:-4, 64:-64, 64:-64])

        shutil.rmtree('/tmp/test')

    def test_blackout_sections(self):
        print('test blackout sections...')
        operator = CutoutOperator(self.volume_path,
                                  mip=self.mip,
                                  blackout_sections=True)

        offset = (4, 64, 64)
        shape = (28, 320, 320)
        output_bbox = Bbox.from_delta(offset, shape)
        chunk = operator(output_bbox)

        img = np.copy(self.img)
        for z in self.blackout_section_ids:
            img[z, :, :] = 0

        img = img[4:-4, 64:-64, 64:-64]
        self.assertTrue(img == chunk)
        shutil.rmtree('/tmp/test')


if __name__ == '__main__':
    unittest.main()
