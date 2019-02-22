#!/usr/bin/env python
import click
from functools import update_wrapper
import numpy as np
import time
import os
import json 
from cloudvolume import CloudVolume, Storage
from cloudvolume.lib import Vec, Bbox

from chunkflow.lib.validate import validate_by_template_matching
from chunkflow.igneous.tasks import downsample_and_upload
from chunkflow.igneous.downsample import downsample_with_averaging
from chunkflow.lib.offset_array import OffsetArray
from chunkflow.aws.cloud_watch import CloudWatch
from chunkflow.aws.sqs_queue import SQSQueue

# import processor functions
from chunkflow.cutout import cutout
from chunkflow.inference import prepare_block_inference_engine, inference
from chunkflow.create_thumbnail import create_thumbnail
from chunkflow.mask import mask

@click.group(chain=True)
@click.option('--mip', type=int, default=0, help='default mip level for all operations.')
@click.option('--show-progress/--not-show-progress', default=False,
              help='show progress bar or not. default is not. ' + 
              'The progress bar should be disabled since google cloud' + 
              'logging will pop out a lot of messages.')
@click.pass_context
def cli(ctx, mip, show_progress):
    """This script processes a chunk in a pipe. 
    One command feeds into the next.
    
    The design is mimicking the click example here:
    https://github.com/pallets/click/blob/master/examples/imagepipe/imagepipe.py
    """
    ctx.ensure_object(dict)
    ctx.obj['mip'] = mip
    ctx.obj['show_progress'] = show_progress


@cli.resultcallback()
def process_commands(processors):
    """This result callback is invoked with an iterable of all the chained subcommands. 
    As in this example each subcommand returns a function we can chain them together to 
    feed one into the other, similar to how a pipe on unix works.
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

def copy_filename(new, old):
    new.filename = old.filename 
    return new

@cli.command('create-task')
@click.option('--queue-name', type=str, default=None, help='sqs queue name')
@click.option('--offset', type=int, nargs=3, default=(0, 0, 0), help='output offset')
@click.option('--shape', type=int, nargs=3, default=(0, 0, 0), help='output shape')
@click.option('--visibility-timeout', type=int, default=None,
              help='visibility timeout of sqs queue; default is using the timeout of the queue.')
@click.pass_context 
@generator 
def create_task_cmd(ctx, queue_name, offset, shape, visibility_timeout):
    """Create task or fetch task from queue."""
    if not queue_name:
        # no queue name specified
        # will only run one task
        stop = np.asarray(offset) + np.asarray(shape)
        output_bbox = Bbox.from_list([*offset, *stop])
        ctx.obj['output_bbox'] = output_bbox 
        yield output_bbox
    else:
        queue = SQSQueue(queue_name, visibility_timeout=visibility_timeout)
        ctx.obj['queue'] = queue
        for task_handle, task in queue:
            print('get task: ', task)
            output_bbox = Bbox.from_filename(task)
            # record the task handle to delete after the processing
            ctx.obj['task_handle'] = task_handle
            ctx.obj['output_bbox'] = output_bbox
            yield output_bbox


@cli.command('delete-task')
@click.pass_context
@processor
def delete_task_cmd(ctx):
    """Delete the task in queue."""
    queue = ctx.obj['queue']
    task_handle = ctx.obj['task_handle']
    queue.delete(task_handle)


@cli.command('cutout')
@click.option('--volume-path', type=str, required=True, help='volume path')
@click.option('--expand-margin-size', type=int, nargs=3, default=(0,0,0), 
              help='include surrounding regions of output bounding box.')
@click.option('--fill-missing/--no-fill-missing', default=False,
              help='fill the missing chunks in input volume with zeros ' + 
              'or not, default is false')
@click.option('--validate-mip', type=int, default=None, help='validate chunk using higher mip level')
@click.pass_context
@processor
def cutout_cmd(ctx, output_bbox, volume_path, expand_margin_size, fill_missing, validate_mip):
    """Cutout chunk from volume."""
    chunk = cutout(chunk_slices, volume_path, output_bbox, mip=ctx.obj['mip'], 
                   show_progress=ctx.obj['show_progress'],
                   expand_margin_size=expand_margin_size,
                   fill_missing=fill_missing, validate_mip=validate_mip)
    yield chunk


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
@click.option('--framework', type=click.Choice(['pznet', 'pytorch', 'pytorch-multitask']), 
              default='pytorch-multitask', help='inference framework')
@click.pass_context
@processor
def inference_cmd(ctx, chunk, convnet_model, convnet_weight_path, patch_size,
              patch_overlap, output_key, original_num_output_channels,
              num_output_channels, framework):
    """Perform convolutional network inference for chunks."""
    if 'block_inference_engine' not in ctx.obj:
        ctx.obj['block_inference_engine'] = prepare_block_inference_engine(
            convnet_model, convnet_weight_path, 
            patch_size=patch_size, patch_overlap=patch_overlap, output_key=output_key,
            original_num_output_channels=original_num_output_channels, 
            num_output_channels=num_output_channels, 
            framework=framework)
    yield inference(chunk, ctx.obj['block_inference_engine'])


@cli.command('create-thumbnail')
@click.option('--volume-path', type=str, required=True, help='thumbnail volume path')
@click.pass_context 
@processor 
def create_thumbnail_cmd(ctx, chunk, volume_path):
    """create quantized thumbnail layer for visualization."""
    chunk_mip = ctx.obj['mip']
    if 'thumbnail_volume' not in ctx.obj['thumbnail_volume']:
        ctx.obj['thumbnail_volume'] = CloudVolume(
            volume_path,
            compress='gzip',
            fill_missing=True,
            bounded=False,
            autocrop=True,
            mip=chunk_mip,
            progress=ctx.obj['show_progress'])
    create_thumbnail(chunk, ctx.obj['thumbnail_volume'], chunk_mip)


@cli.command('mask')
@click.option('--volume-path', type=str, required=True, help='mask volume path')
@click.option('--mask-mip', type=int, default=5, help='mip level of mask')
@click.option('--inverse/--no-inverse', default=True, 
              help='inverse the mask or not. default is True. ' + 
              'the mask will be multiplied to chunk.')
@click.option('--fill-missing/--no-fill-missing', default=False, 
              help='fill missing blocks with black or not. ' + 
              'default is False.')
@click.pass_context
@processor
def mask_cmd(ctx, chunk, volume_path, mask_mip, inverse, fill_missing):
    """Mask the chunk. The mask could be in higher mip level and we
    will automatically upsample it to the same mip level with chunk.
    """
    yield mask(chunk, volume_path, mask_mip, inverse, 
               ctx.obj['chunk_mip'],
               fill_missing=fill_missing, 
               show_progress=ctx.obj['show_progress']) 


@cli.command('crop-margin')
@click.option('--margin-size', type=int, nargs=3, default=None, help='crop the chunk margin.')
@click.pass_context
@processor 
def crop_margin_cmd(ctx, chunk, margin_size):
    """Crop the margin of chunk."""
    if not margin_size:
        print('automatically crop the chunk to output bounding box.')
        yield chunk.cutout(ctx.obj['bbox'].to_slices())
    else:
        if chunk.ndim==3:
            yield chunk[margin_size[0]:chunk.shape[1]-margin_size[0],
                        margin_size[1]:chunk.shape[2]-margin_size[1],
                        margin_size[2]:chunk.shape[3]-margin_size[2]]
        elif chunk.ndim==4:
            yield chunk[:, 
                        margin_size[0]:chunk.shape[1]-margin_size[0],
                        margin_size[1]:chunk.shape[2]-margin_size[1],
                        margin_size[2]:chunk.shape[3]-margin_size[2]]
        else:
            raise ValueError('the array dimension can only by 3 or 4.')


@cli.command('save')
@click.option('--volume-path', type=str, required=True, help='volume path')
@click.pass_context 
@processor 
def save_cmd(ctx, chunk, volume_path):
    """Save chunk to volume."""
    if 'output_volume' not in ctx.obj['output_volume']:
        ctx.obj['output_volume'] = CloudVolume(
            volume_path,
            fill_missing=True,
            bounded=False,
            autocrop=True,
            mip=ctx.obj['mip'],
            progress=ctx.obj['show_progress'])
    
    # transpose czyx to xyzc order
    chunk = np.transpose(chunk)
    ctx.obj['output_volume'][chunk.slices] = chunk
    chunk = np.transpose(chunk)
    yield chunk 


if __name__ == '__main__':
    cli(obj={})

