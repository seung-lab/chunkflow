#!/usr/bin/env python

import click
import numpy as np
import concurrent.futures as futures
import os
import time
import traceback

from cloudvolume import Bbox, EmptyVolumeException
from chunkflow.aws.sqs_queue import SQSQueue
from chunkflow.flow import InferenceExecutor


@click.group()
@click.option(
    '--image-layer-path', type=str, required=True, help='image layer path')
@click.option(
    '--output-layer-path', type=str, required=True, help='output layer path')
@click.option(
    '--output-offset',
    type=int,
    nargs=3,
    default=(0, 0, 0),
    help='output offset')
@click.option(
    '--output-shape',
    type=int,
    nargs=3,
    default=(0, 0, 0),
    help='output shape')
@click.option('--queue-name', type=str, default=None, help='sqs queue name')
@click.option(
    '--cropping-margin-size',
    type=int,
    nargs=3,
    default=(4, 64, 64),
    help='cropping size')
@click.option(
    '--mip', type=int, default=0, help='mip level of image and output')
@click.option(
    '--image-mask-layer-path',
    type=str,
    default=None,
    help='image mask layer path')
@click.option(
    '--output-mask-layer-path',
    type=str,
    default=None,
    help='output mask layer path')
@click.option(
    '--image-mask-mip',
    type=int,
    default=None,
    help='mip level of output mask')
@click.option(
    '--output-mask-mip',
    type=int,
    default=None,
    help='mip level of output mask')
@click.option(
    '--fill-image-missing/--not-fill-image-missing',
    default=False,
    help=
    'fill the missing chunks in image volume with zeros or not, default is false'
)
@click.option(
    '--inverse-image-mask/--not-inverse-image-mask',
    default=True,
    help='inverse the image mask or not.')
@click.option(
    '--inverse-output-mask/--not-inverse-output-mask',
    default=True,
    help='inverse the output mask or not.')
@click.option(
    '--framework',
    type=click.Choice(['pznet', 'pytorch', 'pytorch-multitask']),
    default='pytorch-multitask',
    help='inference framework')
@click.option(
    '--missing-section_ids_file_name',
    type=str,
    default=None,
    help='black out the missing sections recorded in a txt file.' +
    'the section id is simply a list of z coordinates of missing sections')
@click.option(
    '--image-validate-mip',
    type=int,
    default=None,
    help='validate image using mip level')
@click.option(
    '--visibility-timeout',
    type=int,
    default=None,
    help='visibility timeout of sqs queue')
@click.option(
    '--proc-num',
    type=int,
    default=1,
    help='number of processes. if set <=0, will equal to the number of cores.')
@click.option(
    '--interval',
    type=int,
    default=0,
    help='interval of processes start time (sec)')
@click.option(
    '--show-progress/--not-show-progress',
    default=False,
    help=
    'show progress bar or not. default is not. The progress bar should be disabled '
    + 'since google cloud logging will pop out a lot of messages.')
@click.pass_context
def chunkflow(ctx, image_layer_path, output_layer_path, output_offset,
              output_shape, queue_name, cropping_margin_size, mip,
              image_mask_layer_path, output_mask_layer_path, image_mask_mip,
              output_mask_mip, fill_image_missing, inverse_image_mask,
              inverse_output_mask, missing_section_ids_file_name,
              image_validate_mip, visibility_timeout, proc_num, interval,
              show_progress):
    ctx.ensure_object(dict)
    ctx.obj['image_layer_path'] = image_layer_path
    ctx.obj['output_layer_path'] = output_layer_path
    ctx.obj['output_offset'] = output_offset
    ctx.obj['output_shape'] = output_shape
    ctx.obj['queue_name'] = queue_name
    ctx.obj['cropping_margin_size'] = cropping_margin_size
    ctx.obj['mip'] = mip
    ctx.obj['image_mask_layer_path'] = image_mask_layer_path
    ctx.obj['output_mask_layer_path'] = output_mask_layer_path
    ctx.obj['image_mask_mip'] = image_mask_mip
    ctx.obj['output_mask_mip'] = output_mask_mip
    ctx.obj['fill_image_missing'] = fill_image_missing
    ctx.obj['inverse_image_mask'] = inverse_image_mask
    ctx.obj['inverse_output_mask'] = inverse_output_mask
    ctx.obj['missing_section_ids_file_name'] = missing_section_ids_file_name
    ctx.obj['image_validate_mip'] = image_validate_mip
    ctx.obj['visibility_timeout'] = visibility_timeout
    ctx.obj['proc_num'] = proc_num
    ctx.obj['interval'] = interval
    ctx.obj['show_progress'] = show_progress


@click.command()
@click.option(
    '--convnet-model',
    type=str,
    required=True,
    help='convnet model path or type.')
@click.option(
    '--convnet-weight-path',
    type=str,
    required=True,
    help='convnet weight path')
@click.option(
    '--patch-size', type=int, nargs=3, required=True, help='patch size')
@click.option(
    '--patch-overlap',
    type=int,
    nargs=3,
    default=(4, 64, 64),
    help='patch overlap')
@click.option(
    '--output-key',
    type=str,
    default='affinity',
    help='key name of output dict')
@click.option(
    '--original-num-output-channels',
    type=int,
    default=3,
    help=
    'number of original output channels. The net could be trained with more than'
    +
    ' final output channels, such as other neighboring edges in affinity map to enhance '
    + 'net generalization capability.')
@click.option(
    '--num-output-channels',
    type=int,
    default=3,
    help='number of output channels')
@click.pass_context
def inference(ctx, convnet_model, convnet_weight_path, patch_size,
              patch_overlap, output_key, original_num_output_channels,
              num_output_channels, mip, framework):

    executor = InferenceExecutor(
        ctx.obj['image_layer_path'],
        ctx.obj['output_layer_path'],
        convnet_model,
        convnet_weight_path,
        patch_size,
        patch_overlap,
        cropping_margin_size=ctx.obj['cropping_margin_size'],
        output_key=output_key,
        original_num_output_channels=original_num_output_channels,
        num_output_channels=num_output_channels,
        mip=ctx.obj['mip'],
        image_mask_layer_path=ctx.obj['image_mask_layer_path'],
        output_mask_layer_path=ctx.obj['output_mask_layer_path'],
        image_mask_mip=ctx.obj['image_mask_mip'],
        output_mask_mip=ctx.obj['output_mask_mip'],
        fill_image_missing=ctx.obj['fill_image_missing'],
        inverse_image_mask=ctx.obj['inverse_image_mask'],
        inverse_output_mask=ctx.obj['inverse_output_mask'],
        framework=framework,
        missing_section_ids_file_name=ctx.obj['missing_section_ids_file_name'],
        image_validate_mip=ctx.obj['image_validate_mip'],
        show_progress=ctx.obj['show_progress'])

    execute(executor, ctx.obj['queue_name'], ctx.obj['output_offset'],
            ctx.obj['output_shape'], ctx.obj['visibility_timeout'],
            ctx.obj['proc_num'], ctx.obj['interval'])


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
            executor(output_bbox)
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


if __name__ == '__main__':
    chunkflow(obj={})
