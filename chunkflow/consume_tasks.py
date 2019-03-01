#!/usr/bin/env python
import os
import click
from functools import update_wrapper
from time import time
import numpy as np
from cloudvolume.lib import Bbox

from chunkflow.aws.sqs_queue import SQSQueue

# import processor functions
from chunkflow.cutout import cutout
from chunkflow.inference import inference
from chunkflow.create_thumbnail import create_thumbnail
from chunkflow.mask import mask
from chunkflow.crop_margin import crop_margin
from chunkflow.save import save
from chunkflow.upload_log import upload_log
from chunkflow.view import view
from chunkflow.create_chunk import create_chunk
from chunkflow.read_file import read_file
from chunkflow.write_h5 import write_h5
from chunkflow.neuroglancer_view import neuroglancer_view


def default_none(ctx, param, value):
    """
    click currently can not use None with tuple type 
    it will return an empty tuple if the default=None 
    details:
    https://github.com/pallets/click/issues/789
    """
    if not value:
        return None


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
    if stream:
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


@cli.command('create-chunk')
@click.option('--size', type=int, nargs=3, default=(64, 64, 64),
              help='the size of created chunk')
@click.option('--dtype', type=str, default='uint8',
              help='the data type of chunk')
@click.option('--voxel-offset', type=int, nargs=3, default=(0,0,0),
              help='offset in voxel number.')
@generator
def create_chunk_cmd(size, dtype, voxel_offset):
    chunk = create_chunk(size=size, dtype=dtype, voxel_offset=voxel_offset)
    yield {'chunk': chunk}


@cli.command('read-file')
@click.option('--file-name', type=str, required=True,
              help='read chunk from file, support .h5 and .tif')
@click.option('--offset', type=int, nargs=3, callback=default_none,
              help='global offset of this chunk')
@generator
def read_file_cmd(file_name, offset):
    chunk = read_file(file_name, global_offset=offset)
    yield {'chunk': chunk}


@cli.command('write-h5')
@click.option('--file-name', type=str, required=True,
              help='file name of hdf5 file, the extention should be .h5')
@processor
def write_h5_cmd(tasks, file_name):
    for task in tasks:
        write_h5(task['chunk'], file_name)
        # keep the pipeline going
        yield task


@cli.command('generate-task')
@click.option('--queue-name', type=str, default=None, help='sqs queue name')
@click.option('--offset', type=int, nargs=3, default=(0, 0, 0), help='output offset')
@click.option('--shape', type=int, nargs=3, default=(0, 0, 0), help='output shape')
@click.option('--visibility-timeout', type=int, default=None,
              help='visibility timeout of sqs queue; default is using the timeout of the queue.')
@click.option('--verbose/--quiet', default=True, help='default is quiet. ' + 
              'This option will be used in other operations.') 
@generator 
def generate_task_cmd(queue_name, offset, shape, visibility_timeout, verbose):
    """Create task or fetch task from queue."""
    task = {
        'verbose': verbose,
        'log': {'timer': {}}
    }
    if not queue_name:
        # no queue name specified
        # will only run one task
        stop = np.asarray(offset) + np.asarray(shape)
        output_bbox = Bbox.from_list([*offset, *stop])
        task['output_bbox'] = output_bbox
        task['log']['output_bbox'] = output_bbox.to_filename()
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
            task['log']['output_bbox'] = output_bbox.to_filename()
            yield task


@cli.command('delete-task-in-queue')
@processor
def delete_task_in_queue_cmd(tasks):
    """Delete the task in queue."""
    for task in tasks:
        queue = task['queue']
        task_handle = task['task_handle']
        queue.delete(task_handle)
        if task['verbose']:
            print('deleted task {} in queue: {}'.format(task_handle, queue))


@cli.command('cutout')
@click.option('--volume-path', type=str, required=True, help='volume path')
@click.option('--mip', type=int, default=None, help='mip level of the cutout.')
@click.option('--expand-margin-size', type=int, nargs=3, default=(0,0,0), 
              help='include surrounding regions of output bounding box.')
@click.option('--fill-missing/--no-fill-missing', default=False,
              help='fill the missing chunks in input volume with zeros ' + 
              'or not, default is false')
@click.option('--validate-mip', type=int, default=None, help='validate chunk using higher mip level')
@processor
def cutout_cmd(tasks, volume_path, mip, expand_margin_size, fill_missing, validate_mip):
    """Cutout chunk from volume."""
    for task in tasks:
        if mip is None:
            mip = task['mip']
        if 'mip' not in task:
            # set up default mip
            task['mip'] = mip
        start = time()
        task['chunk'] = cutout(task['output_bbox'], 
                               volume_path, mip=task['mip'],
                               expand_margin_size=expand_margin_size,
                               verbose=task['verbose'], 
                               fill_missing=fill_missing, 
                               validate_mip=validate_mip)
        task['log']['timer']['cutout'] = time() - start
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
        start = time()
        task['chunk'] = inference(
            task['chunk'],
            convnet_model, convnet_weight_path, 
            patch_size=patch_size, patch_overlap=patch_overlap, output_key=output_key,
            original_num_output_channels=original_num_output_channels, 
            num_output_channels=num_output_channels, 
            framework=framework,
            log = task['log'],
            verbose=task['verbose']
        )
        task['log']['timer']['inference'] = time() - start 
        yield task


@cli.command('create-thumbnail')
@click.option('--volume-path', type=str, default=None, help='thumbnail volume path')
@processor 
def create_thumbnail_cmd(tasks, volume_path):
    """create quantized thumbnail layer for visualization. 
    Note that the float data type will be quantized to uint8.
    """
    for task in tasks:
        if not volume_path:
            volume_path = os.path.join(task['output_volume_path'],
                                       'thumbnail')
        start = time()
        create_thumbnail(task['chunk'], volume_path, task['mip'],
                         verbose=task['verbose'])
        task['log']['timer']['create-thumbnail'] = time() - start
        yield task


@cli.command('mask')
@click.option('--volume-path', type=str, required=True, help='mask volume path')
@click.option('--mask-mip', type=int, default=5, help='mip level of mask')
@click.option('--inverse/--no-inverse', default=False, 
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
        start = time()
        task['chunk'] = mask(task['chunk'], 
                             volume_path, mask_mip, task['mip'], 
                             fill_missing=fill_missing, 
                             inverse=inverse,
                             verbose=task['verbose'])
        # Note that mask operation could be used several times, 
        # this will only record the last masking operation 
        task['log']['timer']['mask'] = time() - start
        yield task


@cli.command('crop-margin')
@click.option('--margin-size', type=int, nargs=3, default=None,
              help='crop the chunk margin. ' + 
              'The default is None and will use the output_bbox ' + 
              'as croping range.')
@processor
def crop_margin_cmd(tasks, margin_size):
    """Crop the margin of chunk."""
    for task in tasks:
        start = time()
        task['chunk'] = crop_margin(task['chunk'], 
                                    output_bbox=task['output_bbox'],
                                    margin_size=margin_size, 
                                    verbose=task['verbose'])
        task['log']['timer']['crop-margin'] = time() - start
        yield task


@cli.command('save')
@click.option('--volume-path', type=str, required=True, help='volume path')
@processor 
def save_cmd(tasks, volume_path):
    """Save chunk to volume."""
    for task in tasks:
        start = time()
        save(task['chunk'], volume_path, task['mip'], 
             verbose=task['verbose'])
        task['output_volume_path'] = volume_path
        task['log']['timer']['save'] = time() - start
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
        upload_log(log_path, task['log'], task['output_bbox'],
                   verbose=task['verbose'])
        yield task


@cli.command('view')
@processor
def view_cmd(tasks):
    for task in tasks:
        view(task['chunk'])
        yield task


@cli.command('neuroglancer')
@click.option('--voxel-size', nargs=3, type=int, default=(1,1,1),
              help='voxel size of chunk')
@processor
def neuroglancer_cmd(tasks, voxel_size):
    for task in tasks:
        neuroglancer_view([task['chunk'],], voxel_size=voxel_size)
        yield task


if __name__ == '__main__':
    cli()

