import click 
from cloudvolume import Bbox
import boto3 
from tqdm import tqdm 
import numpy as np
import itertools

@click.command()
@click.option('--start', '-s',type=int, default=(0,0,0), nargs=3, help='start of the output')
@click.option('--block-size', '-b', type=int, required=True, nargs=3, help='size/shape of output blocks')
@click.option('--grid-size', '-g', type=int, default=(1,1,1), nargs=3, help='grid size of output blocks')
@click.option('--queue-name', '-q', type=str, default='chunkflow', help='sqs queue name')


def command(start, block_size, grid_size, queue_name):
    start = np.asarray(start)
    block_size = np.asarray(block_size)

    client = boto3.client('sqs')
    resp = client.get_queue_url(QueueName=queue_name)
    queue_url = resp['QueueUrl']
	
    task_entries = []
    for (z,y,x) in tqdm(itertools.product(range(grid_size[0]), range(grid_size[1]), range(grid_size[2]))):
        block_start = start + np.asarray((z,y,x)) * block_size 
        block_stop = block_start + block_size
        bbox = Bbox.from_list([*block_start, *block_stop]) 
        task = bbox.to_filename()
        entry = {
            'Id': task,
            'MessageBody': task
        }
        task_entries.append(entry)

        # use batch mode to produce tasks
        if len(task_entries) == 10:
            resp = client.send_message_batch(QueueUrl=queue_url, Entries=task_entries)
            # the failed list should be empty
            assert 'Failed' not in resp
            task_entries.clear()
        
    # send the remaining tasks less than 10 
    if task_entries:
        resp = client.send_message_batch(QueueUrl=queue_url, Entries=task_entries)
        assert 'Failed' not in resp
        task_entries.clear()


if __name__ == '__main__':
    command()

