import unittest

from .cloud_watch import CloudWatch 


class TestCloudWatch(unittest.TestCase):
    def setUp(self):
        print('test cloud watch...')
        log_name = 'chunkflow-test'
        self.cloud_watch = CloudWatch(log_name)

    def test_put_metric_data(self):
        log = {
            'compute_device': 'X86-64',
            'timer':{
                'cutout': 24,
                'inference': 200,
                'save': 50
            }
        }
        self.cloud_watch.put_metric_data(log)

if __name__ == '__main__':
    unittest.main()
