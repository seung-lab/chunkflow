#!/usr/bin/env python
import os
import click
from functools import update_wrapper
from time import time
import numpy as np
from cloudvolume.lib import Bbox

from chunkflow.aws.sqs_queue import SQSQueue

# import operator functions
from .operators import *


# global dict to hold the operators and parameters
state = {
    'operators': {}
}


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
@click.option('--verbose/--quiet', default=True, 
              help='print informations or not, default is quiet.')
@click.option('--mip', type=int, default=0, help='default mip level of chunks.')
# the code design is from:
# https://github.com/pallets/click/blob/master/examples/imagepipe/imagepipe.py
def cli(verbose, mip):
    """This script processes a chunk in a pipe. 
    One command feeds into the next.
    """
    state['verbose'] = verbose
    state['mip'] = mip
    pass


@cli.resultcallback()
def process_commands(operators, verbose, mip):
    """This result callback is invoked with an iterable of all 
    the chained subcommands. As in this example each subcommand 
    returns a function we can chain them together to feed one 
    into the other, similar to how a pipe on unix works.
    """
    # Start with an empty iterable 
    stream = ()

    # Pipe it through all stream operators.
    for operator in operators:
        stream = operator(stream)

    # Evaluate the stream and throw away the items.
    if stream:
        for _ in stream:
            pass


def operator(f):
    """Help decorator to rewrite a function so that it returns another function from it."""
    def new_func(*args, **kwargs):
        def operator(stream):
            return f(stream, *args, **kwargs)
        return operator
    return update_wrapper(new_func, f)


def generator(f):
    """Similar to the :func:`operator` but passes through old values unchanged and does not pass 
    through the values as parameter.
    """
    @operator
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
    """[generator] Create a fake chunk for easy test."""
    create_chunk_operator = CreateChunkOperator()
    chunk = create_chunk_operator(size=size, dtype=dtype, voxel_offset=voxel_offset)
    yield {'chunk': chunk}


@cli.command('read-file')
@click.option('--file-name', type=str, required=True,
              help='read chunk from file, support .h5 and .tif')
@click.option('--offset', type=int, nargs=3, callback=default_none,
              help='global offset of this chunk')
@generator
def read_file_cmd(file_name, offset):
    """[generator] Read HDF5 and tiff files."""
    read_file_operator = ReadFileOperator()
    chunk = read_file_operator(file_name, global_offset=offset)
    yield {'chunk': chunk}


@cli.command('write-h5')
@click.option('--name', type=str, default='write-h5', help='name of this operator')
@click.option('--file-name', type=str, required=True,
              help='file name of hdf5 file, the extention should be .h5')
@operator
def write_h5_cmd(tasks, name, file_name):
    """[operator] Write chunk to HDF5 file."""
    state['operators'][name] = WriteH5Operator()
    for task in tasks:
        state['operators'][name](task['chunk'], file_name)
        # keep the pipeline going
        yield task


@cli.command('generate-task')
@click.option('--queue-name', type=str, default=None, help='sqs queue name')
@click.option('--offset', type=int, nargs=3, default=(0, 0, 0), help='output offset')
@click.option('--shape', type=int, nargs=3, default=(0, 0, 0), help='output shape')
@click.option('--visibility-timeout', type=int, default=None,
              help='visibility timeout of sqs queue; default is using the timeout of the queue.')
@generator 
def generate_task_cmd(queue_name, offset, shape, visibility_timeout):
    """[generator] Create task or fetch task from queue."""
    task = {'log': {'timer': {}}}
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
@click.option('--name', type=str, default='delete-task-in-queue', help='name of this operator')
@operator
def delete_task_in_queue_cmd(tasks, name):
    """[operator] Delete the task in queue."""
    for task in tasks:
        queue = task['queue']
        task_handle = task['task_handle']
        queue.delete(task_handle)
        if state['verbose']:
            print('deleted task {} in queue: {}'.format(task_handle, queue))


@cli.command('cutout')
@click.option('--name', type=str, default='cutout', help='name of this operator')
@click.option('--volume-path', type=str, required=True, help='volume path')
@click.option('--mip', type=int, default=None, help='mip level of the cutout.')
@click.option('--expand-margin-size', type=int, nargs=3, default=(0,0,0), 
              help='include surrounding regions of output bounding box.')
@click.option('--fill-missing/--no-fill-missing', default=False,
              help='fill the missing chunks in input volume with zeros ' + 
              'or not, default is false')
@click.option('--validate-mip', type=int, default=None, 
              help='validate chunk using higher mip level')
@click.option('--blackout-sections/--no-blackout-sections', 
              default=False, help='blackout some sections. ' +
              'the section ids json file should named blackout_section_ids.json.' +
              'default is False.')
@operator
def cutout_cmd(tasks, name, volume_path, mip, expand_margin_size, 
               fill_missing, validate_mip, blackout_sections):
    """[operator] Cutout chunk from volume."""
    state['operators'][name] = CutoutOperator(
        volume_path, mip=state['mip'], 
        expand_margin_size=expand_margin_size,
        verbose=state['verbose'], fill_missing=fill_missing,
        validate_mip=validate_mip, blackout_sections=blackout_sections,
        name=name)
 
    for task in tasks:
        start = time()
        task['chunk'] = state['operators'][name](task['output_bbox'])
        task['log']['timer'][name] = time() - start
        task['cutout_volume_path'] = volume_path
        yield task


@cli.command('normalize-section-contrast')
@click.option('--name', type=str, default='normalize-section-contrast', help='name of operator.')
@click.option('--levels-path', type=str, default=None, 
              help='the path of section histograms.')
@click.option('--mip', type=int, default=None, 
              help='the mip level of section histograms.')
@click.option('--clip-fraction', type=float, default=0.01, 
              help='the voxel intensity fraction to clip out.')
@click.option('--minval', type=float, default=None,
              help='the minimum intensity of transformed chunk.')
@click.option('--maxval', type=float, default=None,
              help='the maximum intensity of transformed chunk.')
@operator
def normalize_contrast_contrast_cmd(tasks, name, levels_path, mip, clip_fraction,
                           minval, maxval):
    """[operator] Normalize the section contrast using precomputed histograms."""
    if mip is None:
        mip = state['mip']
    if levels_path is None:
        levels_path = state['cutout_volume_path']

    state['operators'][name] = NormalizeSectionContrastOperator(
        levels_path, mip, clip_fraction, minval=minval, maxval=maxval, name=name)
    
    for task in tasks:
        start = time()
        task['chunk'] = state['operators'][name](task['chunk'])
        task['log']['timer'][name] = time() - start
        yield task


@cli.command('inference')
@click.option('--name', type=str, default='inference', help='name of this operator')
@click.option('--convnet-model', type=str, default=None, help='convnet model path or type.')
@click.option('--convnet-weight-path', type=str, default=None, help='convnet weight path')
@click.option('--patch-size', type=int, nargs=3, default=(20, 256, 256), help='patch size')
@click.option('--patch-overlap', type=int, nargs=3, default=(4, 64, 64), help='patch overlap')
@click.option('--output-key', type=str, default='affinity', help='key name of output dict')
@click.option('--original-num-output-channels', type=int, default=3,
              help='number of original output channels. The net could be trained with more than' +
              ' final output channels, such as other neighboring edges in affinity map to enhance ' +
              'net generalization capability.')
@click.option('--num-output-channels', type=int, default=3, help='number of output channels')
@click.option('--framework', type=click.Choice(['identity', 'pznet', 'pytorch', 'pytorch-multitask']), 
              default='pytorch-multitask', help='inference framework')
@click.option('--batch-size', type=int, default=1, help='mini batch size of input patch.')
@operator
def inference_cmd(tasks, name, convnet_model, convnet_weight_path, patch_size,
              patch_overlap, output_key, original_num_output_channels,
              num_output_channels, framework, batch_size):
    """[operator] Perform convolutional network inference for chunks."""
    state['operators'][name] = InferenceOperator(
        convnet_model, convnet_weight_path, 
        patch_size=patch_size, output_key=output_key,
        num_output_channels=num_output_channels,
        original_num_output_channels=original_num_output_channels,
        patch_overlap=patch_overlap,
        framework=framework,
        batch_size=batch_size,
        verbose=state['verbose'], name=name)

    for task in tasks:
        if 'log' not in task:
            task['log'] = {'timer': {}}
        start = time()
        task['chunk'] = state['operators'][name](task['chunk'])
        task['log']['timer'][name] = time() - start
        task['log']['compute_device'] = state['operators'][name].compute_device
        yield task


@cli.command('mask')
@click.option('--name', type=str, default='mask', help='name of this operator')
@click.option('--volume-path', type=str, required=True, help='mask volume path')
@click.option('--mip', type=int, default=5, help='mip level of mask')
@click.option('--inverse/--no-inverse', default=False, 
              help='inverse the mask or not. default is True. ' + 
              'the mask will be multiplied to chunk.')
@click.option('--fill-missing/--no-fill-missing', default=False, 
              help='fill missing blocks with black or not. ' + 
              'default is False.')
@operator
def mask_cmd(tasks, name, volume_path, mip, inverse, fill_missing):
    """[operator] Mask the chunk. The mask could be in higher mip level and we
    will automatically upsample it to the same mip level with chunk.
    """
    state['operators'][name] = MaskOperator(volume_path, mip, state['mip'], 
                                 inverse=inverse,
                                 fill_missing=fill_missing,
                                 verbose=state['verbose'],
                                 name=name)

    for task in tasks:
        start = time()
        task['chunk'] = state['operators'][name](task['chunk']) 
        # Note that mask operation could be used several times, 
        # this will only record the last masking operation 
        task['log']['timer'][name] = time() - start
        yield task


@cli.command('crop-margin')
@click.option('--name', type=str, default='crop-margin', help='name of this operator')
@click.option('--margin-size', type=int, nargs=3, default=None,
              help='crop the chunk margin. ' + 
              'The default is None and will use the output_bbox ' + 
              'as croping range.')
@operator
def crop_margin_cmd(tasks, name, margin_size):
    """[operator] Crop the margin of chunk."""
    state['operators'][name] = CropMarginOperator(margin_size=margin_size, 
                                       verbose=state['verbose'],
                                       name=name)
    for task in tasks:
        start = time()
        task['chunk'] = state['operators'][name](task['chunk'], 
                                      output_bbox=task['output_bbox'])
        task['log']['timer'][name] = time() - start
        yield task


@cli.command('save')
@click.option('--name', type=str, default='save', help='name of this operator')
@click.option('--volume-path', type=str, required=True, help='volume path')
@click.option('--upload-log/--no-upload-log', default=True, 
              help='the log will be put inside volume-path')
@click.option('--nproc', type=int, default=0,
              help='number of processes, negative means using all the cores, ' +
              '0/1 means turning off multiple processing, ' +
              'n>1 means using n processes')
@click.option('--create-thumbnail/--no-create-thumbnail', default=False,
              help='create thumbnail or not. '+
              'the thumbnail is a downsampled and quantized version of the chunk.')
@operator 
def save_cmd(tasks, name, volume_path, upload_log, nproc, create_thumbnail):
    """[operator] Save chunk to volume."""
    state['operators'][name] = SaveOperator(
        volume_path, state['mip'], upload_log=upload_log,
        create_thumbnail=create_thumbnail, nproc=nproc,
        verbose=state['verbose'], name=name)

    for task in tasks:
        # the time elapsed was recorded internally
        state['operators'][name](
            task['chunk'], log=task.get('log', None), 
            output_bbox=task.get('output_bbox', None))
        task['output_volume_path'] = volume_path
        yield task


@cli.command('cloud-watch')
@click.option('--name', type=str, default='cloud-watch', help='name of this operator')
@click.option('--log-name', type=str, default='chunkflow', help='name of the speedometer')
@operator
def cloud_watch_cmd(tasks, name, log_name):
    """[operator] Real time speedometer in AWS CloudWatch."""
    state['operators'][name]=CloudWatchOperator(log_name=log_name, name=name,
                                                verbose=state['verbose'])
    for task in tasks:
        state['operators'][name](task['log'])
        yield task


@cli.command('view')
@click.option('--name', type=str, default='view', help='name of this operator')
@operator
def view_cmd(tasks, name):
    """[operator] Visualize the chunk using cloudvolume view in browser."""
    state['operators'][name] = ViewOperator(name=name)
    for task in tasks:
        state['operators'][name](task['chunk'])
        yield task


@cli.command('neuroglancer')
@click.option('--name', type=str, default='neuroglancer', help='name of this operator')
@click.option('--voxel-size', nargs=3, type=int, default=(1,1,1),
              help='voxel size of chunk')
@operator
def neuroglancer_cmd(tasks, name, voxel_size):
    """[operator] Visualize the chunk using neuroglancer."""
    state['operators'][name] = NeuroglancerViewOperator(name=name)
    for task in tasks:
        state['operators'][name]([task['chunk'],], voxel_size=voxel_size)
        yield task


if __name__ == '__main__':
    cli()

