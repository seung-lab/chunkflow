import os
import time
import traceback
import concurrent.futures as futures
import numpy as np
from cloudvolume import Bbox, EmptyVolumeException

from chunkflow.aws.sqs_queue import SQSQueue


def execute(executor, queue_name, output_offset, output_shape,
            visibility_timeout, proc_num, interval):
    if not queue_name:
        # no queue name specified
        # will only run one task
        output_stop = np.asarray(output_offset) + np.asarray(output_shape)
        output_bbox = Bbox.from_list([*output_offset, *output_stop])
        executor(output_bbox)
    else:
        if proc_num <= 0:
            # use all the cores!
            proc_num = os.cpu_count()
        if proc_num == 1:
            process_queue(
                executor, queue_name, visibility_timeout=visibility_timeout)
        else:
            print('launching {} processes.'.format(proc_num))
            with futures.ProcessPoolExecutor(
                    max_workers=proc_num) as pool_executor:
                for i in range(proc_num):
                    pool_executor.submit(
                        process_queue,
                        executor,
                        queue_name,
                        sleep_time=i * interval,
                        visibility_timeout=visibility_timeout)


def process_queue(executor, queue_name, sleep_time=0, visibility_timeout=None):
    print('sleep for {} seconds and then start working...'.format(sleep_time))
    time.sleep(sleep_time)
    # queue name was defined, read from sqs queue
    queue = SQSQueue(queue_name, visibility_timeout=visibility_timeout)

    try:
        for task_handle, task in queue:
            print('get task: ', task)
            output_bbox = Bbox.from_filename(task)
            yield(output_bbox)
            queue.delete(task_handle)
    except EmptyVolumeException:
        print('raised an EmptyVolumeException, please check the ' +
              'volume whether there is some chunk file missing.')
        raise
    except TypeError as err:
        print('get TypeError: {}'.format(err))
        print('probably because the queue becomes None somehow.')
        print('continue working...')
        raise
        #process_queue(executor, queue_name, sleep_time=sleep_time,
        #              visibility_timeout=visibility_timeout)
    except Exception as err:
        print(task, ' raised {}\n {}'.format(err, traceback.format_exc()))
        raise

