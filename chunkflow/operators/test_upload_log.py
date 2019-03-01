import unittest
from .upload_log import UploadLogOperator
from cloudvolume.lib import Bbox


class TestUploadLog(unittest.TestCase):
    def test_upload_log(self):
        output_bbox = Bbox.from_list([0, 0, 0, 3, 3, 3]) 
        log = {
            'compute_device': 'x86-64',
            'output_bbox': output_bbox.to_filename(),
            'timer': {
                'inference': 100
            }
        }

        operator = UploadLogOperator('file:///tmp/log/test/')
        operator(log, output_bbox)
