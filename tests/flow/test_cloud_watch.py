import unittest
from chunkflow.flow.cloud_watch import CloudWatchOperator


class TestUploadLog(unittest.TestCase):
    def test_upload_log(self):
        log = {'compute_device': 'x86-64', 'timer': {'inference': 100}}
        operator = CloudWatchOperator()
        operator(log)
