from .operator_base import OperatorBase
import json
from cloudvolume import Storage
from chunkflow.aws.cloud_watch import CloudWatch


class UploadLogOperator(OperatorBase):
    def __init__(self, log_path, verbose=True, name='upload-log'):
        super().__init__(name=name)
        self.log_path = log_path
        self.verbose = verbose
       
        # write to aws cloud watch
        self.aws_cloud_watch = CloudWatch('chunkflow')
        self.storage = Storage(log_path)

    def __call__(self, log, output_bbox):
        if self.verbose:
            print('save log {} to {}'.format(output_bbox.to_filename(), 
                                             self.log_path))
 
        self.aws_cloud_watch.put_metric_data(log)

        log_text = json.dumps(log)
        
        # write to google cloud storage 
        self.storage.put_file(
            file_path=output_bbox.to_filename() + '.json',
            content=log_text,
            content_type='application/json')
