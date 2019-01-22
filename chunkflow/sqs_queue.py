import boto3 


class SQSQueue(object):
    def __init__(self, queue_name, visibility_timeout=1800):
        self.client = boto3.client('sqs')
        resp = self.client.get_queue_url(QueueName=queue_name)
        self.queue_url = resp['QueueUrl']
        self.visibility_timeout = visibility_timeout

    def __iter__(self):
        return self 

    def __next__(self):
        resp = self.client.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=1,
            MessageAttributeNames=['All'],
            VisibilityTimeout=self.visibility_timeout,
            # we should set this wait time to use long poll
            # checkout the AWS documentation here:
            # https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-long-polling.html#sqs-short-long-polling-differences
            WaitTimeSeconds=20
        )
        if 'Messages' not in resp:
            return None 
        else:
            return resp['Messages'][0]['Body']
    
    def _send_entry_list(self, entry_list):
        resp = self.client.send_message_batch(QueueUrl=self.queue_url, Entries=entry_list)
        # the failed list should be empty
        assert 'Failed' not in resp
        entry_list.clear()
         
    def send_message_list(self, message_list):
        '''
        the messages are string 
        use batch mode to send the messages quickly 
        '''
        # the maximum number in a batch is 10 
        task_entries=[]
        for message in message_list:
            entry = {
                'Id': message,
                'MessageBody': message
            }
            task_entries.append(entry)

            # use batch mode to produce tasks
            if len(task_entries) == 10:
                self._send_entry_list(task_entries)
            
        # send the remaining tasks less than 10 
        if task_entries:
            self._send_entry_list(task_entries)

        # make sure that the remaining task list is empty
        assert not task_entries

     
