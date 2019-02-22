import json
from cloudvolume import Storage
from .aws.cloud_watch import CloudWatch


def upload_log(log_path, log, output_bbox):
    # write to aws cloud watch
    aws_cloud_watch = CloudWatch('inference')
    aws_cloud_watch.put_metric_data(log)

    log_text = json.dumps(log)
    
    # write to google cloud storage 
    with Storage(log_path) as storage:
        storage.put_file(
            file_path=output_bbox.to_filename() + '.json',
            content=log_text,
            content_type='application/json')
