import json
from cloudvolume import Storage

from .operator_base import OperatorBase


class UploadLogOperator(OperatorBase):
    def __init__(self, log_path: str, verbose: bool=True, name: str='upload-log'):
        super().__init__(name=name)
        self.log_path = log_path
        self.verbose = verbose
        self.storage = Storage(log_path)

    def __call__(self, log, output_bbox):
        if self.verbose:
            print('save log {} to {}'.format(output_bbox.to_filename(),
                                             self.log_path))
        # write to google cloud storage 
        self.storage.put_file(
            file_path=output_bbox.to_filename() + '.json',
            content=json.dumps(log),
            content_type='application/json')
