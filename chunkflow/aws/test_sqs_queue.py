from .sqs_queue import SQSQueue
import unittest


class TestSQSQueue(unittest.TestCase):
    def setUp(self):
        queue_name = 'chunkflow-test'
        print('test sqs queue using queue of {}'.format(queue_name))
        # use small fetch_wait_time_seconds to make the test faster 
        # we should use 20 seconds in production run to use long pull
        # see more details in the class documentation.
        self.queue = SQSQueue(queue_name, wait_if_empty=None,
                              fetch_wait_time_seconds=1)

    def test_send_and_receive_message_list(self):
        print('start sending messages...')
        message_list = []
        for i in range(23):
            message_list.append(str(i))
        self.queue.send_message_list(message_list)

        print('start receiving messages in iteration...')
        i = 0
        for receipt_handle, message in self.queue:
            print('received message: {}'.format(i))
            self.assertTrue(int(message) in range(23))
            i += 1
            self.queue.delete(receipt_handle)


if __name__ == '__main__':
    unittest.main()
