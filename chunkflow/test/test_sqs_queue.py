from chunkflow.sqs_queue import SQSQueue 
import unittest


class TestSQSQueue(unittest.TestCase):
    def setUp(self):
        self.queue = SQSQueue('chunkflow-test')

    def test_send_message_list(self):
        message_list = []
        for i in range(23):
            message_list.append(str(i))
        self.queue.send_message_list(message_list)

    def test_iteration(self):
        i = 0
        for message in self.queue:
            if not message:
                # get None, there is no more messages
                break
            self.assertTrue( int(message) in range(23)) 
            i+=1
       

if __name__ == '__main__':
    unittest.main()
