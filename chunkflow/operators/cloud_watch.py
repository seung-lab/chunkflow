from .operator_base import OperatorBase
from chunkflow.aws.cloud_watch import CloudWatch


class CloudWatchOperator(OperatorBase):
    def __init__(self, log_name='chunkflow', name='cloud-watch'):
        super().__init__(name=name)

        # write to aws cloud watch
        self.aws_cloud_watch = CloudWatch(log_name)
    
    def __call__(self, log):
        self.aws_cloud_watch.put_metric_data(log)
