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
        for receipt_handle, message in self.queue:
            print(i)
            self.assertTrue(int(message) in range(23))
            i += 1
            self.queue.delete(receipt_handle)


if __name__ == '__main__':
    unittest.main()
