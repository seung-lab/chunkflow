#!/usr/bin/env python
import os
import click
from functools import update_wrapper
import numpy as np
from cloudvolume.lib import Bbox

from chunkflow.aws.sqs_queue import SQSQueue

# import processor functions
from chunkflow.cutout import cutout
from chunkflow.inference import inference
from chunkflow.create_thumbnail import create_thumbnail
from chunkflow.mask import mask
from chunkflow.save import save
from chunkflow.upload_log import upload_log 


@click.group(chain=True)
def cli():
    """This script processes a chunk in a pipe. 
    One command feeds into the next.
    
    The design is mimicking the click example here:
    https://github.com/pallets/click/blob/master/examples/imagepipe/imagepipe.py
    """


@cli.resultcallback()
def process_commands(processors):
    """This result callback is invoked with an iterable of all 
    the chained subcommands. As in this example each subcommand 
    returns a function we can chain them together to feed one 
    into the other, similar to how a pipe on unix works.
    """
    # Start with an empty iterable 
    stream = ()

    # Pipe it through all stream processors.
    for processor in processors:
        stream = processor(stream)

    # Evaluate the stream and throw away the items.
    for _ in stream:
        pass


def processor(f):
    """Help decorator to rewrite a function so that it returns another function from it."""
    def new_func(*args, **kwargs):
        def processor(stream):
            return f(stream, *args, **kwargs)
        return processor
    return update_wrapper(new_func, f)


def generator(f):
    """Similar to the :func:`processor` but passes through old values unchanged and does not pass 
    through the values as parameter.
    """
    @processor
    def new_func(stream, *args, **kwargs):
        for item in stream:
            yield item
        for item in f(*args, **kwargs):
            yield item 
    return update_wrapper(new_func, f)


@cli.command('generate-task')
@click.option('--mip', type=int, default=0, help='default mip level for all operations.')
@click.option('--show-progress/--not-show-progress', default=False,
              help='show progress bar or not. default is not. ' + 
              'The progress bar should be disabled since google cloud' + 
              'logging will pop out a lot of messages.')
@click.option('--queue-name', type=str, default=None, help='sqs queue name')
@click.option('--offset', type=int, nargs=3, default=(0, 0, 0), help='output offset')
@click.option('--shape', type=int, nargs=3, default=(0, 0, 0), help='output shape')
@click.option('--visibility-timeout', type=int, default=None,
              help='visibility timeout of sqs queue; default is using the timeout of the queue.')
@generator 
def generate_task_cmd(mip, show_progress, queue_name, offset, shape, visibility_timeout):
    """Create task or fetch task from queue."""
    task = {
        'mip': mip,
        'show_progress': show_progress
    }
    if not queue_name:
        # no queue name specified
        # will only run one task
        stop = np.asarray(offset) + np.asarray(shape)
        output_bbox = Bbox.from_list([*offset, *stop])
        task['output_bbox'] = output_bbox 
        yield task 
    else:
        queue = SQSQueue(queue_name, visibility_timeout=visibility_timeout)
        task['queue'] = queue
        for task_handle, output_bbox_str in queue:
            print('get task: ', output_bbox_str)
            output_bbox = Bbox.from_filename(output_bbox_str)
            # record the task handle to delete after the processing
            task['task_handle'] = task_handle
            task['output_bbox'] = output_bbox
            yield task


@cli.command('delete-task-in-queue')
@processor
def delete_task_in_queue_cmd(tasks):
    """Delete the task in queue."""
    for task in tasks:
        queue = task['queue']
        task_handle = task['task_handle']
        queue.delete(task_handle)


@cli.command('cutout')
@click.option('--volume-path', type=str, required=True, help='volume path')
@click.option('--expand-margin-size', type=int, nargs=3, default=(0,0,0), 
              help='include surrounding regions of output bounding box.')
@click.option('--fill-missing/--no-fill-missing', default=False,
              help='fill the missing chunks in input volume with zeros ' + 
              'or not, default is false')
@click.option('--validate-mip', type=int, default=None, help='validate chunk using higher mip level')
@processor
def cutout_cmd(tasks, volume_path, expand_margin_size, fill_missing, validate_mip):
    """Cutout chunk from volume."""
    for task in tasks:
        task['chunk'] = cutout(task['output_bbox'], 
                               volume_path, mip=task['mip'],
                               expand_margin_size=expand_margin_size,
                               show_progress=task['show_progress'], 
                               fill_missing=fill_missing, 
                               validate_mip=validate_mip)
        yield task


@cli.command('inference')
@click.option('--convnet-model', type=str, required=True, help='convnet model path or type.')
@click.option('--convnet-weight-path', type=str, required=True, help='convnet weight path')
@click.option('--patch-size', type=int, nargs=3, required=True, help='patch size')
@click.option('--patch-overlap', type=int, nargs=3, default=(4, 64, 64), help='patch overlap')
@click.option('--output-key', type=str, default='affinity', help='key name of output dict')
@click.option('--original-num-output-channels', type=int, default=3,
              help='number of original output channels. The net could be trained with more than' +
              ' final output channels, such as other neighboring edges in affinity map to enhance ' +
              'net generalization capability.')
@click.option('--num-output-channels', type=int, default=3, help='number of output channels')
@click.option('--framework', type=click.Choice(['identity', 'pznet', 'pytorch', 'pytorch-multitask']), 
              default='pytorch-multitask', help='inference framework')
@processor
def inference_cmd(tasks, convnet_model, convnet_weight_path, patch_size,
              patch_overlap, output_key, original_num_output_channels,
              num_output_channels, framework):
    """Perform convolutional network inference for chunks."""
    for task in tasks:
        task['chunk'] = inference(
            task['chunk'],
            convnet_model, convnet_weight_path, 
            patch_size=patch_size, patch_overlap=patch_overlap, output_key=output_key,
            original_num_output_channels=original_num_output_channels, 
            num_output_channels=num_output_channels, 
            framework=framework,
            show_progress=task['show_progress']
        )
        yield task


@cli.command('create-thumbnail')
@click.option('--volume-path', type=str, required=True, help='thumbnail volume path')
@processor 
def create_thumbnail_cmd(tasks, volume_path):
    """create quantized thumbnail layer for visualization."""
    for task in tasks:
        create_thumbnail(task['chunk'], volume_path, task['mip'],
                         show_progress=task['show_progress'])
        yield task


@cli.command('mask')
@click.option('--volume-path', type=str, required=True, help='mask volume path')
@click.option('--mask-mip', type=int, default=5, help='mip level of mask')
@click.option('--inverse/--no-inverse', default=True, 
              help='inverse the mask or not. default is True. ' + 
              'the mask will be multiplied to chunk.')
@click.option('--fill-missing/--no-fill-missing', default=False, 
              help='fill missing blocks with black or not. ' + 
              'default is False.')
@processor
def mask_cmd(tasks, volume_path, mask_mip, inverse, fill_missing):
    """Mask the chunk. The mask could be in higher mip level and we
    will automatically upsample it to the same mip level with chunk.
    """
    for task in tasks:
        task['chunk'] = mask(task['chunk'], 
                             volume_path, mask_mip, inverse, 
                             task['mip'], fill_missing=fill_missing, 
                             show_progress=task['show_progress']) 
        yield task


@cli.command('crop-margin')
@click.option('--margin-size', type=int, nargs=3, default=None,
              help='crop the chunk margin.')
@processor
def crop_margin_cmd(tasks, margin_size):
    """Crop the margin of chunk."""
    for task in tasks:
        if not margin_size:
            print('automatically crop the chunk to output bounding box.')
            task['chunk'] = task['chunk'].cutout(
                task['output_bbox'].to_slices())
            yield task
        else:
            chunk = task['chunk']
            if chunk.ndim==3:
                chunk = chunk[margin_size[0]:chunk.shape[1]-margin_size[0],
                              margin_size[1]:chunk.shape[2]-margin_size[1],
                              margin_size[2]:chunk.shape[3]-margin_size[2]]
            elif chunk.ndim==4:
                chunk = chunk[:, 
                              margin_size[0]:chunk.shape[1]-margin_size[0],
                              margin_size[1]:chunk.shape[2]-margin_size[1],
                              margin_size[2]:chunk.shape[3]-margin_size[2]]
            else:
                raise ValueError('the array dimension can only by 3 or 4.')
            task['chunk'] = chunk
            yield task


@cli.command('save')
@click.option('--volume-path', type=str, required=True, help='volume path')
@processor 
def save_cmd(tasks, volume_path):
    """Save chunk to volume."""
    for task in tasks:
        save(task['chunk'], volume_path, task['mip'], 
             show_progress=task['show_progress'])
        task['output_volume_path'] = volume_path 
        yield task


@cli.command('upload-log')
@click.option('--log-path', type=str, default=None, 
              help='log storage path')
@processor
def upload_log_cmd(tasks, log_path):
    for task in tasks:
        if not log_path:
            print('put logs inside output path.')
            log_path = os.path.join(task['output_volume_path'], 'log')

        upload_log(log_path, task['log'], task['output_bbox'])
        yield task


if __name__ == '__main__':
    cli()

