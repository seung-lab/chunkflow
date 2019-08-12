#!/bin/env python

import click 
import boto3 
from tqdm import tqdm 
import numpy as np
from itertools import product

from cloudvolume import Bbox
from chunkflow.lib.aws.sqs_queue import SQSQueue


@click.command()
@click.option('--start', '-s',type=int, default=(0,0,0), nargs=3, help='(z y x), start of the output')
@click.option('--block-size', '-b', type=int, required=True, nargs=3, help='(z y x), size/shape of output blocks')
@click.option('--grid-size', '-g', type=int, default=(1,1,1), nargs=3, help='(z y x), grid size of output blocks')
@click.option('--queue-name', '-q', type=str, default='chunkflow', help='sqs queue name')
def main(start, block_size, grid_size, queue_name):
    start = np.asarray(start)
    block_size = np.asarray(block_size)

    queue = SQSQueue(queue_name)

    tasks = []
    for (z, y, x) in tqdm(product(range(grid_size[0]),
                                  range(grid_size[1]),
                                  range(grid_size[2]))):

        block_start = start + np.asarray((z, y, x)) * block_size
        block_stop = block_start + block_size
        bbox = Bbox.from_list([*block_start, *block_stop])
        task = bbox.to_filename()
        tasks.append(task)

        # use batch mode to produce tasks
        if len(tasks) == 10:
            queue.send_message_list(tasks)
            tasks.clear()
      
    # send the remaining tasks less than 10
    if tasks:
        queue.send_message_list(tasks)
        tasks.clear()


if __name__ == '__main__':
    main()
