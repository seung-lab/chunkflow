from chunkflow.lib.aws.cloud_watch import CloudWatch

from .base import OperatorBase


class CloudWatchOperator(OperatorBase):
    """CloudWatchOperator

    watch the performance of each operator using AWS CloudWatch.

    :param log_name: the log used in AWS CloudWatch
    :return: the timer result will be uploaded to AWS CloudWatch 
        and nothing will return here.
    """
    def __init__(self,
                 log_name: str = 'chunkflow',
                 name: str = 'cloud-watch',
                 verbose: bool = True):
        super().__init__(name=name, verbose=verbose)

        # write to aws cloud watch
        self.aws_cloud_watch = CloudWatch(log_name)

    def __call__(self, log):
        self.aws_cloud_watch.put_metric_data(log)
        if self.verbose:
            print('log submitted to cloud watch: ', log)
