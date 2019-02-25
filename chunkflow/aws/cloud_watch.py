import boto3
from cloudvolume.secrets import aws_credentials


class CloudWatch:
    def __init__(self, log_name):
        self.log_name = log_name
        credentials = aws_credentials()
        self.client = boto3.client(
            'cloudwatch',
            region_name=credentials['AWS_DEFAULT_REGION'],
            aws_secret_access_key=credentials['AWS_SECRET_ACCESS_KEY'],
            aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'])

    def put_metric_data(self, log):
        assert isinstance(log, dict)
        # create metric data
        metric_data = []
        for key, value in log['timer'].items():
            metric_data.append(
                {
                    'MetricName': key,
                    'Dimensions': [
                        {
                            'Name': 'compute_device',
                            'Value': log['compute_device']
                        }
                    ],
                    'Value': value,
                    'Unit': 'Seconds'
                }
            )
        
        # submit the metric data
        self.client.put_metric_data(
            Namespace=self.log_name,
            MetricData=metric_data
        )
