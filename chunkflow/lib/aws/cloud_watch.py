from warnings import warn
import boto3
from cloudvolume.secrets import aws_credentials


class CloudWatch:
    """monitor time elapsed of each operator using AWS CloudWatch."""
    def __init__(self, log_name: str, credentials: dict = None):
        """
        Parameters
        ----------
        log_name: 
            the name of the log.
        credentials: 
            aws credential. If it is None, we'll try to construct it automatically.
        """
        self.log_name = log_name
        if not credentials:
            credentials = aws_credentials()
        self.client = boto3.client(
            'cloudwatch',
            region_name=credentials['AWS_DEFAULT_REGION'],
            aws_secret_access_key=credentials['AWS_SECRET_ACCESS_KEY'],
            aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'])

    def put_metric_data(self, log: dict):
        """
        Parameters
        -----------
        log: 
            the log containing the operator time elapsed in the `timer` key.
        """
        assert isinstance(log, dict)

        if 'compute_device' in log:
            compute_device = log['compute_device']
        else:
            warn(
                'did not find compute device in log, will create one based on CPU.'
            )
            import platform
            compute_device = platform.processor()

        # create metric data
        metric_data = []
        for key, value in log['timer'].items():
            metric_data.append({
                'MetricName':
                key,
                'Dimensions': [{
                    'Name': 'compute_device',
                    'Value': compute_device
                }],
                'Value':
                value,
                'Unit':
                'Seconds'
            })

        # submit the metric data
        self.client.put_metric_data(Namespace=self.log_name,
                                    MetricData=metric_data)
