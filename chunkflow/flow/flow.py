#!/usr/bin/env python
import os
from pathlib import Path
from time import time

from typing import Generator, List

from copy import deepcopy

import numpy as np
import click
import json
from tqdm import tqdm

import zarr
import tinybrain

from chunkflow.lib.flow import *
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec
from cloudfiles import CloudFiles

from chunkflow.lib.aws.sqs_queue import SQSQueue
from chunkflow.lib.cartesian_coordinate import Cartesian, BoundingBox, BoundingBoxes
from chunkflow.lib.synapses import Synapses

from chunkflow.chunk import Chunk
from chunkflow.chunk.image import Image
from chunkflow.chunk.affinity_map import AffinityMap
from chunkflow.chunk.segmentation import Segmentation
from chunkflow.chunk.image.convnet.inferencer import Inferencer
from chunkflow.point_cloud import PointCloud

# import operator functions
from .aggregate_skeleton_fragments import AggregateSkeletonFragmentsOperator
from .cloud_watch import CloudWatchOperator
from .load_precomputed import LoadPrecomputedOperator
from .downsample_upload import DownsampleUploadOperator
from .log_summary import load_log, print_log_statistics
from .mask import MaskOperator
from .mesh import MeshOperator
from .mesh_manifest import MeshManifestOperator
from .neuroglancer import NeuroglancerOperator
from .plugin import Plugin
from .load_pngs import load_png_images
from .save_precomputed import SavePrecomputedOperator
from .save_pngs import SavePNGsOperator
from .setup_env import setup_environment
from .skeletonize import SkeletonizeOperator
from .view import ViewOperator

@main.command('create-bbox')
@click.option('--start', '-s', 
    type=click.INT, required=True, nargs=3,
    help = 'voxel offset or start of the bounding box.')
@click.option('--stop', '-p',
    type=click.INT, default=None, nargs=3, callback=default_none,
    help='voxel stop or end of bounding box.')
@click.option('--size', '-z', 
    type=click.INT, default=None, nargs=3, callback=default_none,
    help='volume size or dimension.')
@generator
def create_bbox(start: tuple, stop: tuple, size: tuple):
    assert stop is not None or size is not None
    if stop is None:
        stop = Cartesian.from_collection(start) + Cartesian.from_collection(size)
    bbox = BoundingBox(start, stop)
    task = get_initial_task()
    task['bbox'] = bbox
    yield task


@main.command('generate-tasks')
@click.option('--layer-path', '-l',
              type=str, default=None,
              help='dataset layer path to fetch dataset information.')
@click.option('--mip', '-m',
              type=click.INT, default=None, help='mip level of the dataset layer.')
@click.option('--roi-start', '-s',
              type=click.INT, default=None, nargs=3, callback=default_none, 
              help='(z y x), start of the chunks')
@click.option('--roi-stop', '-r',
              type=click.INT, nargs=3, default=None, callback=default_none,
              help='stop coordinate of region of interest')
@click.option('--roi-size', '-z',
              type=click.INT, nargs=3, default=None, callback=default_none,
              help='size of region of interest')
@click.option('--chunk-size', '-c',
              type=click.INT, default=None, nargs=3,
              help='(z y x), size/shape of chunks')
@click.option('--bounding-box', '-b', type=str, default=None,
    help='the string representation of a bounding box')
@click.option('--grid-size', '-g',
              type=click.INT, default=None, nargs=3, callback=default_none,
              help='(z y x), grid size of output blocks')
@click.option('--file-path', '-f', default = None,
              type=click.Path(writable=True, dir_okay=False, resolve_path=True),
              help='output tasks as an numpy array formated as npy.')
@click.option('--queue-name', '-q',
              type=str, default=None, help='sqs queue name')
@click.option('--respect-chunk-size/--respect-stop',
              default=True, help="""for the last bounding box, \
make the chunk size consistent or cut off at the stopping boundary.""")
@click.option('--aligned-block-size', '-a',
    type=click.INT, default=None, nargs=3, callback=default_none,
    help='force alignment of block size. Note that the alignment start from (0, 0, 0).')
@click.option('--task-index-start', '-i',
              type=click.INT, default=0, help='starting index of task list.')
@click.option('--task-index-stop', '-p',
              type=click.INT, default=None, help='stop index of task list.')
@click.option('--disbatch/--no-disbatch', '-d',
              default=False, help='use disBatch environment variable or not')
@click.option('--use-https/--use-credential', default=False,
    help='if we read from a public dataset in cloud storage, it is required to use https.')
@generator
def generate_tasks(
        layer_path: str, mip: int, roi_start: tuple, roi_stop: tuple, 
        roi_size: tuple, chunk_size: tuple, bounding_box:str,
        grid_size: tuple, file_path: str, queue_name: str, 
        respect_chunk_size: bool, aligned_block_size: tuple, 
        task_index_start: tuple, task_index_stop: tuple, 
        disbatch: bool, use_https: bool):
    """Generate a batch of tasks."""
    if mip is None:
        mip = state['mip']
    assert mip >=0 

    if bounding_box is not None:
        bboxes = [BoundingBox.from_string(bounding_box)]
        if chunk_size is None:
            chunk_size = bboxes[0].shape
        else:
            assert chunk_size == bboxes[0].shape
    else:
        bboxes = BoundingBoxes.from_manual_setup(
            chunk_size, layer_path=layer_path,
            roi_start=roi_start, roi_stop=roi_stop, 
            roi_size=roi_size, mip=mip, grid_size=grid_size,
            respect_chunk_size=respect_chunk_size,
            aligned_block_size=aligned_block_size,
            use_https=use_https
        )
    print(f'number of all the candidate tasks: {len(bboxes)}')
    
    if task_index_start:
        if task_index_stop is None:
            # task_index_stop = task_index_start + 1
            task_index_stop = len(bboxes)
        bboxes = [*bboxes[task_index_start:task_index_stop]]
        logging.info(f'selected task indexes from {task_index_start} to {task_index_stop}')
    elif disbatch:
        assert 'DISBATCH_REPEAT_INDEX' in os.environ
        disbatch_index = int(os.environ['DISBATCH_REPEAT_INDEX'])
        assert disbatch_index < len(bboxes), f'DISBATCH_REPEAT_INDEX is larger than the task number!'
        bboxes = [bboxes[disbatch_index],]
        logging.info(f'selected a task with disBatch index {disbatch_index}')
        
    # write out as a file
    # this could be used for iteration in slurm cluster.
    if file_path:
        if not file_path.endswith('.npy'):
            file_path += len(bboxes) + '.npy'
        bboxes.to_file(file_path)

    # if state['verbose']:
    bbox_num = len(bboxes)
    logging.info(f'total number of tasks: {bbox_num}') 
    print(f'total number of tasks: {bbox_num}') 

    if queue_name is not None:
        queue = SQSQueue(queue_name)
        queue.send_message_list(bboxes)
    else:
        for bbox_index, bbox in enumerate(bboxes):
            if disbatch:
                assert len(bboxes) == 1
                bbox_index = disbatch_index
            print(f'executing task {bbox_index+task_index_start} in {bbox_num+task_index_start} with bounding box: {bbox.string}')
            logging.info(f'executing task {bbox_index+task_index_start} in {bbox_num+task_index_start} with bounding box: {bbox.string}')
            task = get_initial_task()
            task['bbox'] = bbox
            task['bbox_index'] = bbox_index
            task['bbox_num'] = bbox_num
            task['log']['bbox'] = bbox.string
            yield task


@main.command('adjust-bbox')
@click.option('--corner-offset', '-c', type=click.INT, nargs=6, default=None,
    help='adjust bounding box corner offset')
@operator
def adjust_bbox(tasks, corner_offset: tuple):
    """adjust the corner of bounding box."""
    for task in tasks:
        if task is not None:
            bbox = task['bbox']
            bbox = bbox.adjust_corner(corner_offset)
            logging.info(f'after bounding box adjustment: {bbox.string}')
            task['bbox'] = bbox
        yield task


@main.command('skip-task')
@click.option('--prefix', '-p', required=True, type=str,
    help='the pre part of result file path')
@click.option('--suffix', '-s', required=True, type=str,
    help='the post part of result file path. Normally include file extention.')
@click.option('--mode', '-m', 
    type=click.Choice(['missing', 'empty', 'exist']), default='exist',
    help='skip this task if the corresponding file is missing/empty/exists')
@click.option('--adjust-size', '-a', default=None, type=click.INT, callback=default_none,
    help='expand or shrink the bounding box. Currently, cloud-volume Bbox only support symetric grow.')
@operator
def skip_task(tasks: Generator, prefix: str, suffix: str, 
        mode: str, adjust_size: int):
    """if a result file already exists, skip this task."""
    for task in tasks:
        if task is not None:
            bbox = task['bbox']
            if adjust_size is not None:
                bbox = bbox.clone()
                bbox = bbox.adjust(adjust_size)
            file_name = prefix + bbox.string + suffix

            if 'empty' in mode:
                if not os.path.exists(file_name) or os.path.getsize(file_name)==0:
                    logging.info(f'file {file_name} do not exist or is empty, skip this task.')
                    task = None
            elif 'missing' in mode:
                if not os.path.exists(file_name):
                    logging.info(f'the file {file_name} is missing, skip this task')
                    task = None
            elif 'exist' in mode:
                if os.path.exists(file_name):
                    logging.info(f'the file {file_name} already exist, skip this task')
                    task = None
            
        yield task


@main.command('mark-complete')
@click.option('--prefix', '-p', type=str, default=None,
    help='pre-path of a file.')
@click.option('--suffix', '-s', type=str, default=None,
    help='suffix of the flag file.')
@operator
def mark_complete(tasks, prefix: str, suffix: str):
    """mark completion of a task as an empty file.

    Args:
        tasks (stream): the tasks stream
        prefix (str): prefix of the mark file
        suffix (str): suffix of the mark file

    Yields:
        None
    """
    for task in tasks:
        if task is not None:
            bbox = task['bbox']
            fname = f'{prefix}{bbox.string}{suffix}'
            Path(fname).touch()
        yield task

@main.command('skip-all-zero')
@click.option('--input-chunk-name', '-i',
    type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--prefix', '-p', type=str, default=None, 
    help = 'pre-path of a file. we would like to keep a trace that this task was executed.')
@click.option('--suffix', '-s', type=str, default=".h5",
    help='post-path of a file. normally include the extention of result file.')
@click.option('--adjust-size', '-a', type=click.INT, default=None,
    help='change the bounding box of chunk if it do not match with final result file name.')
@click.option('--chunk-bbox/--task-bbox', default=True,
    help='use the bbox in task or generate from chunk. Default is using chunk bounding box.')
@operator
def skip_all_zero(tasks, input_chunk_name: str, prefix: str, suffix: str, adjust_size: int, chunk_bbox: bool):
    """if chunk has all zero, skip this task."""
    for task in tasks:
        if task is not None:
            chunk = task[input_chunk_name]
            if not np.any(chunk):
                logging.info('all zero chunk, skip this task')
                if prefix is not None or suffix is not None:
                    if chunk_bbox:
                        bbox = chunk.bbox.clone()
                    else:
                        bbox = task['bbox']
                    if adjust_size is not None:
                        bbox = bbox.adjust(adjust_size)
                    fname = f'{prefix}{bbox.string}{suffix}'
                    if not os.path.exists(fname):
                        logging.info(f'create an empty file as mark: {fname}')
                        Path(fname).touch()
                # target task as None and task will be skipped
                task = None
        yield task


@main.command('skip-none')
@click.option('--input-name', '-i',
    type=str, default=DEFAULT_CHUNK_NAME, help='input name')
@click.option('--touch/--no-touch', default=True, help='touch an empty file or not')
@click.option('--prefix', '-p', default=None, help='prefix of output file.')
@click.option('--suffix', '-s', default=None, help='suffix of output file.')
@operator
def skip_none(tasks: dict, input_name: str, touch: bool, prefix: str, suffix: str):
    """If item is None, skip this task."""
    for task in tasks:
        if task is not None:
            data = task[input_name]
            if data is None:
                # target task as None and task will be skipped
                task = None
                if touch:
                    assert prefix is not None
                    assert suffix is not None
                    bbox = task['bbox']
                    fname = f'{prefix}{bbox.string}{suffix}'
                    Path(fname).touch()
        yield task


@main.command('setup-env')
@click.option('--volume-start', required=True, nargs=3, type=click.INT,
              help='start coordinate of output volume in mip 0')
@click.option('--volume-stop', default=None, type=click.INT, nargs=3, callback=default_none,
              help='stop coordinate of output volume (noninclusive like python coordinate) in mip 0.')
@click.option('--volume-size', '-s',
              default=None, type=click.INT, nargs=3, callback=default_none, 
              help='size of output volume.')
@click.option('--layer-path', '-l',
              type=str, required=True, help='the path of output volume.')
@click.option('--max-ram-size', '-r',
              default=15, type=click.INT, help='the maximum ram size (GB) of worker process.')
@click.option('--output-patch-size', '-z',
              type=click.INT, required=True, nargs=3, help='output patch size.')
@click.option('--input-patch-size', '-i',
              type=click.INT, default=None, nargs=3, callback=default_none,
              help='input patch size.')
@click.option('--channel-num', '-c',
              type=click.INT, default=1, 
              help='output patch channel number. It is 3 for affinity map.')
@click.option('--dtype', '-d', type=click.Choice(['uint8', 'float16', 'float32']), 
              default='float32', help='output numerical precision.')
@click.option('--output-patch-overlap', '-o',
              type=click.INT, default=None, nargs=3, callback=default_none,
              help='overlap of patches. default is 50% overlap')
@click.option('--crop-chunk-margin', '-c', 
              type=click.INT, nargs=3, default=None,
              callback=default_none, help='size of margin to be cropped.')
@click.option('--mip', '-m', type=click.IntRange(min=0, max=3), default=0, 
              help='the output mip level (default is 0).')
@click.option('--thumbnail-mip', '-b', type=click.IntRange(min=5, max=16), default=6,
              help='mip level of thumbnail layer.')
@click.option('--max-mip', '-x', type=click.IntRange(min=5, max=16), default=8, 
              help='maximum MIP level for masks.')
@click.option('--queue-name', '-q',
              type=str, default=None, help='sqs queue name.')
@click.option('--visibility-timeout', '-t',
              type=click.INT, default=3600, help='visibility timeout of the AWS SQS queue.')
@click.option('--thumbnail/--no-thumbnail', default=True, help='create thumbnail or not.')
@click.option('--encoding', '-e',
              type=click.Choice(['raw', 'jpeg', 'compressed_segmentation', 
                                 'fpzip', 'kempressed']), default='raw', 
              help='Neuroglancer precomputed block compression algorithm.')
@click.option('--voxel-size', '-v', type=click.INT, nargs=3, default=(40, 4, 4),
              help='voxel size or resolution of mip 0 image.')
@click.option('--overwrite-info/--no-overwrite-info', default=False,
              help='normally we should avoid overwriting info file to avoid errors.')
@generator
def setup_env(volume_start, volume_stop, volume_size, layer_path, 
              max_ram_size, output_patch_size, input_patch_size, channel_num, dtype, 
              output_patch_overlap, crop_chunk_margin, mip, thumbnail_mip, max_mip,
              queue_name, visibility_timeout, thumbnail, encoding, voxel_size, 
              overwrite_info):
    """Setup convolutional net inference environment."""
    bboxes = setup_environment(
        state['dry_run'], volume_start, volume_stop, volume_size, layer_path, 
        max_ram_size, output_patch_size, input_patch_size, channel_num, dtype, 
        output_patch_overlap, crop_chunk_margin, mip, thumbnail_mip, max_mip,
        thumbnail, encoding, voxel_size, overwrite_info)
 
    if queue_name is not None and not state['dry_run']:
        queue = SQSQueue(queue_name, visibility_timeout=visibility_timeout)
        queue.send_message_list(bboxes)
    else:
        for bbox in bboxes:
            task = get_initial_task()
            task['bbox'] = bbox
            task['log']['bbox'] = bbox.string
            yield task


@main.command('cloud-watch')
@click.option('--name',
              type=str,
              default='cloud-watch',
              help='name of this operator')
@click.option('--log-name',
              type=str,
              default='chunkflow',
              help='name of the speedometer')
@operator
def cloud_watch(tasks, name, log_name):
    """Real time speedometer in AWS CloudWatch."""
    operator = CloudWatchOperator(log_name=log_name, name=name)
    for task in tasks:
        if task is not None:
            operator(task['log'])
        yield task


@main.command('cleanup')
@click.option('--dir', '-d',
    default = './',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    help='the files in a directory')
@click.option('--mode', '-m', 
    type=click.Choice(['exist', 'empty', 'not-empty']), default='exist',
    help='condition of clean up.')
@click.option('--suffix', '-s', type=str, default=None,
    help='suffix of files.')
@generator
def cleanup(dir: str, mode: str, suffix: str):
    dir = os.path.expanduser(dir)

    for fname in tqdm(os.listdir(dir), desc='removing files: '):
        fname = os.path.join(dir, fname)
        if suffix and not fname.endswith(suffix):
            # only consider files with matching suffix
            continue
        if 'exist' == mode:
            if os.path.exists(fname):
                os.remove(fname)
        elif 'empty' == mode:
            if os.path.getsize(fname) == 0:
                os.remove(fname)
        elif 'not-empty' == mode:
            if os.path.getsize(fname) > 0:
                os.remove(fname)
        else:
            raise ValueError(f'unsupported mode: {mode}')

    logging.info(f'there are {len(os.listdir(dir))} files remaining.')
    yield None


@main.command('create-info')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME,
              help="create info for this chunk.")
@click.option('--output-layer-path', '-l', type=str, default="file://.", 
              help='path of output layer.')
@click.option('--channel-num', '-c', type=click.INT, default=1, help='number of channel')
@click.option('--layer-type', '-t',
              type=click.Choice(['image', 'segmentation']),
              default=None, help='type of layer. either image or segmentation.')
@click.option('--data-type', '-d',
              type=click.Choice(['uint8', 'uint32', 'uint64', 'float32']),
              default = None, help='data type of array')
@click.option('--encoding', '-e',
              type=click.Choice(['raw', 'jpeg', 'compressed_segmentation', 
                    'kempressed', 'npz', 'fpzip', 'npz_uint8']),
              default='raw', help='compression algorithm.')
@click.option('--voxel-size', '-s', default=None, type=click.INT, nargs=3, callback=default_none,
              help='voxel size with unit of nm')
@click.option('--voxel-offset', '-o', default=None, type=click.INT, nargs=3, callback=default_none,
              help='voxel offset of array')
@click.option('--volume-size', '-v',
              type=click.INT, nargs=3, default=None, callback=default_none,
              help='total size of the volume.')
@click.option('--block-size', '-b',
              type=click.INT, nargs=3, required=True,
              help='chunk size of each file.')
@click.option('--factor', '-f',
              type=click.INT, nargs=3, default=(2,2,2),
              help='hierarchical downsampling factor')
@click.option('--max-mip', '-m',
              type=click.INT, default=0, 
              help = 'maximum mip level.')
@operator
def create_info(tasks,input_chunk_name: str, output_layer_path: str, channel_num: int, 
                layer_type: str, data_type: str, encoding: str, voxel_size: tuple, 
                voxel_offset: tuple, volume_size: tuple, block_size: tuple, factor: tuple, max_mip: int):
    """Create attrsdata for Neuroglancer Precomputed volume."""
    
    for task in tasks:
        if task is not None:
            if not input_chunk_name in task:
                if voxel_offset is None:
                    voxel_offset = Cartesian(0, 0, 0)
            else:
                chunk = task[input_chunk_name]
                if chunk.ndim == 3:
                    channel_num = 1
                elif chunk.ndim == 4:
                    channel_num = chunk.shape[0]
                else:
                    raise ValueError('chunk dimension can only be 3 or 4')

                if voxel_offset is None:
                    voxel_offset = chunk.voxel_offset
                if voxel_size is None:
                    voxel_size = chunk.voxel_size

                volume_size = chunk.shape
                data_type = chunk.dtype.name

                if layer_type is None:
                    if np.issubdtype(chunk.dtype, np.uint8) or \
                            np.issubdtype(chunk.dtype, np.float32) or \
                            np.issubdtype(chunk.dtype, np.float16):
                        layer_type = 'image'
                    else:
                        layer_type = 'segmentation'
            
            assert volume_size is not None
            assert data_type is not None
            if data_type == 'segmentation':
                mesh = "mesh"
            else:
                mesh = None

            info = CloudVolume.create_new_info(
                channel_num, layer_type=layer_type,
                data_type=data_type,
                encoding=encoding,
                resolution=voxel_size[::-1],
                voxel_offset=voxel_offset[::-1],
                volume_size=volume_size[::-1],
                chunk_size=block_size[::-1],
                factor=Vec(factor),
                max_mip=max_mip,
                compressed_segmentation_block_size=(8, 8, 8),
                mesh = mesh,
                )
            vol = CloudVolume(output_layer_path, info=info)
            vol.commit_info()
        yield task


@main.command('fetch-task-from-file')
@click.option('--file-path', '-f',
              type=click.Path(file_okay=True, dir_okay=False, exists=True, 
                              readable=True, resolve_path=True),
              help='file contains bounding boxes or tasks.')
@click.option('--job-index', '-i', 
              type=click.INT, default=None,
              help='index of task in the tasks.')
@click.option('--slurm-job-array/--no-slurm-job-array',
              default=False, help='use the slurm job array '+
              'environment variable to identify task index.')
@click.option('--granularity', '-g',
              type=click.INT, default=1, help='number of tasks to do in one run.')
@generator
def fetch_task_from_file(file_path: str, job_index: int, slurm_job_array: bool, granularity: int):
    """Fetch task from a file containing bounding boxes."""
    if(slurm_job_array):
        job_index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    assert job_index is not None

    bbox_array = np.load(file_path)
    task_start = job_index * granularity 
    task_stop = min(bbox_array.shape[0], task_start + granularity)
    for idx in range(task_start, task_stop):
        bbox = BoundingBox.from_list(bbox_array[idx, :])
        task = get_initial_task()
        task['bbox'] = bbox
        yield task


@main.command('fetch-task-from-sqs')
@click.option('--queue-name', '-q',
                type=str, default=None, help='sqs queue name')
@click.option('--visibility-timeout', '-v',
    type=click.INT, default=None, 
    help='visibility timeout of sqs queue; default is using the timeout of the queue.')
@click.option('--num', '-n', type=click.INT, default=-1,
              help='fetch limited number of tasks.' +
              ' This is useful in local cluster to control task time elapse.' + 
              'Negative value will be infinite.')
@click.option('--retry-times', '-r',
              type=click.INT, default=30,
              help='the times of retrying if the queue is empty.')
@generator
def fetch_task_from_sqs(queue_name, visibility_timeout, num, retry_times):
    """Fetch task from queue."""
    # This operator is actually a generator,
    # it replaces old tasks to a completely new tasks and loop over it!
    queue = SQSQueue(queue_name, 
                     visibility_timeout=visibility_timeout,
                     retry_times=retry_times)
    while num!=0:
        task_handle, bbox_str = queue.handle_and_message
        if task_handle is None:
            return
        num -= 1
        
        logging.info(f'get task: {bbox_str}')
        bbox = BoundingBox.from_filename(bbox_str)
        
        # record the task handle to delete after the processing
        task = get_initial_task() 
        task['queue'] = queue
        task['task_handle'] = task_handle
        task['bbox'] = bbox
        task['log']['bbox'] = bbox.string
        yield task


@main.command('aggregate-skeleton-fragments')
@click.option('--name', type=str, default='aggregate-skeleton-fragments',
              help='name of operator')
@click.option('--input-name', '-i', type=str, default='prefix',
              help='input prefix name in task stream.')
@click.option('--prefix', '-p', type=str, default=None,
              help='prefix of skeleton fragments.')
@click.option('--fragments-path', '-f', type=str, required=True,
              help='storage path of skeleton fragments.')
@click.option('--output-path', '-o', type=str, default=None,
              help='storage path of aggregated skeletons.')
@operator
def aggregate_skeleton_fragments(tasks, name, input_name, prefix, fragments_path, output_path):
    """Merge skeleton fragments."""
    if output_path is None:
        output_path = fragments_path

    operator = AggregateSkeletonFragmentsOperator(fragments_path, output_path)
    if prefix:
        operator(prefix)
    else:
        for task in tasks:
            if task is not None:
                start = time()
                operator(task[input_name])
                task['log']['timer'][name] = time() - start
            yield task


@main.command('create-chunk')
@click.option('--size', '-s', type=click.INT, nargs=3,
    default=Cartesian(64, 64, 64), help='the size of created chunk')
@click.option('--dtype', '-d',
    type=click.Choice(
        ['uint8', 'uint32', 'uint16', 'uint64', 'float32', 'float64']),
    default='uint8', help='the data type of chunk')
@click.option('--pattern', '-p', type=click.Choice(['sin', 'zero', 'random']), 
    default='sin', help='ways to generate array.')
@click.option('--voxel-offset', '-t',
    type=click.INT, nargs=3, default=(0, 0, 0), help='offset in voxel number.')
@click.option('--voxel-size', '-e',
    type=click.INT, nargs=3, default=(1,1,1), help='voxel size in nm')
@click.option('--output-chunk-name', '-o',
    type=str, default="chunk", help="name of created chunk")
@operator
def create_chunk(tasks, size, dtype, pattern, voxel_offset, voxel_size, output_chunk_name):
    """Create a fake chunk for easy test."""
    logging.info(f'creating chunk: {output_chunk_name}')
    for task in tasks:
        if task is not None:
            task[output_chunk_name] = Chunk.create(
                size=size, dtype=np.dtype(dtype),
                pattern=pattern, 
                voxel_offset=voxel_offset,
                voxel_size=voxel_size)
        yield task


@main.command('load-synapses')
@click.option('--name', '-n', type=str, default='load-synapses', 
    help='name of operator')
@click.option('--file-path', '-f',
    type=click.Path(file_okay=True, dir_okay=True, resolve_path=True),
    required=True, 
    help='files containing synapses. Currently support HDF5 and JSON.')
@click.option('--suffix', '-s', type=str, default=None, 
    help='file path suffix.')
@click.option('--c-order/--f-order', default=True,
    help='C order or Fortran order in the file. XYZ is Fortran order, ZYX is C order.')
@click.option('--resolution', '-r', type=click.INT, nargs=3, 
    default=None, callback=default_none, help='resolution of points.')
@click.option('--remove-outside/--keep-all', default=False, 
    help='remove synapses outside of the bounding box or not.')
@click.option('--set-bbox/--not-set-bbox', default=False)
@click.option('--output-name', '-o', type=str, default=DEFAULT_SYNAPSES_NAME,
    help='data name of the result.')
@operator
def load_synapses(tasks, name: str, file_path: str, suffix: str, 
        c_order: bool, resolution: tuple, remove_outside: bool, 
        set_bbox: bool, output_name: str):
    """Load synapses formated as JSON or HDF5."""
    for task in tasks:
        if task is not None:
            start = time()
            if os.path.isfile(file_path) and \
                    (file_path.endswith('.h5') or \
                    file_path.endswith('.json')):
                fname = file_path
            elif os.path.isdir(file_path):
                bbox = task['bbox']
                if suffix is not None:
                    fname = os.path.join(file_path, f'{bbox.string}{suffix}')
                else:
                    fname = os.path.join(file_path, f'{bbox.string}')
                    if not os.path.exists(fname) and '.' not in fname:
                        fname += '.h5'
                        
            elif not os.path.exists(file_path):
                bbox = task['bbox']
                fname = f'{file_path}{bbox.string}{suffix}'
            else:
                fname = file_path
            assert os.path.isfile(fname), f'can not find file: {fname}'

            if set_bbox:
                bbox = BoundingBox.from_string(fname)
                assert bbox is not None
                task['bbox'] = bbox

            if os.path.getsize(fname) == 0:
                task[output_name] = None
            else:
                syns = Synapses.from_file(
                    fname, 
                    resolution = resolution,
                    c_order = c_order
                )
                if remove_outside:
                    bbox = task['bbox']
                    syns.remove_synapses_outside_bounding_box(bbox)
                logging.info(f'loaded synapses with {syns.pre_num} presynapses and {syns.post_num} post synapses.')
                if syns.pre_num == 0:
                    syns = None
                task[output_name] = syns
            task['log']['timer'][name] = time() - start
        yield task


@main.command('save-points')
@click.option('--input-name', '-i', type=str, default='point_cloud')
@click.option('--file-path', '-f',
    type=click.Path(file_okay=True, dir_okay=True, resolve_path=True),
    required=True, help='HDF5 file path.')
@operator
def save_points(tasks, input_name: str, file_path: str):
    """Save synapses as HDF5 file."""
    for task in tasks:
        if task is not None:
            points = task[input_name]
            assert isinstance(points, PointCloud)
            points.to_h5(file_path)
        yield task

 
@main.command('save-synapses')
@click.option('--input-name', '-i', type=str, default=DEFAULT_SYNAPSES_NAME)
@click.option('--file-path', '-f',
    type=click.Path(file_okay=True, dir_okay=True, resolve_path=True),
    required=True, help='HDF5 file path.')
@operator
def save_synapses(tasks, input_name: str, file_path: str):
    """Save synapses as HDF5 file."""
    for task in tasks:
        if task is not None:
            syns = task[input_name]
            if file_path.endswith('.json'):
                data = syns.json_dict
                with open(file_path, 'w') as file:
                    json.dump(data, file)
            else:
                if not file_path.endswith('.h5'):
                    if 'bbox' in task:
                        bbox = task['bbox']
                        if os.path.isdir(file_path):
                            file_path = os.path.join(file_path, bbox.string)
                        else:
                            file_path += bbox.string
                    file_path += '.h5'
                if syns is None:
                    Path(file_path).touch()
                else:
                    syns.to_h5(file_path)
        yield task

@main.command('load-npy')
@click.option('--name', '-n', type=str, default='load-npy', help='name of operator')
@click.option('--file-path', '-f', 
    type=click.Path(file_okay=True, dir_okay=True, resolve_path=True),
    required=True, help='NPY file path')
@click.option('--resolution', '-r', type=click.FLOAT, nargs=3, default=(1,1,1), help='resolution of points.')
@click.option('--output-name', '-o', type=str, default='array', help='data name of the result.')
@operator
def read_npy(tasks, name: str, file_path: str, resolution: tuple, output_name: str):
    """Read NPY files."""
    for task in tasks:
        if task is not None:
            start = time()
            if not file_path.endswith('.npy'):
                bbox = task['bbox']
                if os.path.isdir(file_path):
                    file_path = os.path.join(file_path, f'{bbox.string}.npy')
                else:
                    file_path = f'{file_path}{bbox.string}.npy'
            assert os.path.exists(file_path)
            if 0 == os.path.getsize(file_path):
                task[output_name] = None
            else:
                with open(file_path, 'rb') as file:
                    array = np.load(file)
                array *= np.asarray(resolution, dtype=array.dtype)
                task[output_name] = array
            task['log']['timer'][name] = time() - start
        yield task

@main.command('load-json')
@click.option('--name', '-n', type=str, default='load-json', help='name of operator.')
@click.option('--file-path', '-f', 
    type=click.Path(file_okay=True, dir_okay=True, resolve_path=True), 
    required=True, help='JSON file name')
@click.option('--output-name', '-o', type=str, default='dict', help='data name of the result.')
@operator
def read_json(tasks, name: str, file_path: str, output_name: str):
    """Read JSON file."""
    for task in tasks:
        if task is not None:
            start = time()
            if not file_path.endswith('.json'):
                bbox = task['bbox']
                if os.path.isdir(file_path):
                    file_path = os.path.join(file_path, f'{bbox.string}.json')
                else:
                    file_path = f'{file_path}{bbox.string}.json'
            assert os.path.exists(file_path)
            with open(file_path, 'r') as file:
                task[output_name] = json.load(file)
            task['log']['timer'][name] = time() - start
        yield task


@main.command('load-nrrd')
@click.option('--name', type=str, default='load-nrrd',
              help='read nrrd file from local disk.')
@click.option('--file-name', '-f', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='read chunk from NRRD file')
@click.option('--voxel-offset', '-v', type=click.INT, nargs=3, default=None, callback=default_none,
              help='global offset of this chunk')
@click.option('--voxel-size', '-s', type=click.INT, nargs=3, default=None, callback=default_none,
              help='physical size of voxels. The unit is assumed to be nm.')
@click.option('--dtype', '-d',
              type=click.Choice(['uint8', 'uint32', 'uint64', 'float32', 'float64', 'float16']),
              help='convert to data type')
@click.option('--output-chunk-name', '-o', type=str, default='chunk',
              help='chunk name in the global state')
@operator
def read_nrrd(tasks, name: str, file_name: str, voxel_offset: tuple,
             voxel_size: tuple, dtype: str, output_chunk_name: str):
    """Read NRRD file."""
    for task in tasks:
        if task is not None:
            start = time()
            task[output_chunk_name] = Chunk.from_nrrd(
                file_name,
                dtype=dtype,
                voxel_offset=voxel_offset,
                voxel_size=voxel_size)
            task['log']['timer'][name] = time() - start
        yield task


@main.command('save-nrrd')
@click.option('--name', type=str, default='save-nrrd', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--file-name', '-f', default=None,
    type=click.Path(dir_okay=False, resolve_path=True), 
    help='file name of NRRD file.')
@operator
def save_nrrd(tasks, name, input_chunk_name, file_name):
    """Save chunk as a NRRD file."""
    for task in tasks:
        if task is not None:
            task[input_chunk_name].to_nrrd(file_name)
        yield task


@main.command('load-pngs')
@click.option('--path-prefix', '-p',
              required=True, type=str,
              help='directory path prefix of png files.')
@click.option('--output-chunk-name', '-o',
              type=str, default=DEFAULT_CHUNK_NAME,
              help='output chunk name')
@click.option('--cutout-offset', '-c',
              type=click.INT, default=(0,0,0), nargs=3,
              help='cutout chunk from an offset')
@click.option('--voxel-offset', '-t',
              type=click.INT, nargs=3, default=(0,0,0),
              help = 'the offset of png images volume, could be negative.')
@click.option('--voxel-size', '-x', type=click.INT, nargs=3, default=(1,1,1), callback=default_none,
              help='physical size of voxels. the unit is assumed to be nm.')
@click.option('--digit-num', '-d', type=click.INT, default=5,
    help='the total number of digits with leading zero padding. For example, digit_num=3, the file name will be like 001.png')
@click.option('--chunk-size', '-s',
              type=click.INT, nargs=3, default=None, callback=default_none,
              help='cutout chunk size')
@click.option('--dtype', type=str, default='uint8', 
    help = 'data type of output chunk.')
@operator
def load_pngs(tasks: dict, path_prefix: str, 
                output_chunk_name: str, cutout_offset: tuple,
                voxel_offset: tuple, voxel_size: tuple, 
                digit_num: int, chunk_size: tuple, dtype: str):
    """Read a serials of png files."""
    cutout_offset = Cartesian.from_collection(cutout_offset)
    voxel_offset = Cartesian.from_collection(voxel_offset)
    voxel_size = Cartesian.from_collection(voxel_size)
    for task in tasks:
        if task is not None:
            if chunk_size is None:
                if 'bbox' in task:
                    bbox = task['bbox']
                else:
                    bbox = None
            else:
                bbox = BoundingBox.from_delta(cutout_offset, chunk_size)

            task[output_chunk_name] = load_png_images(
                path_prefix, bbox = bbox, 
                voxel_offset=voxel_offset,
                digit_num=digit_num,
                voxel_size=voxel_size,
                dtype=dtype)
        yield task


@main.command('load-tif')
@click.option('--name', type=str, default='load-tif',
              help='read tif file from local disk.')
@click.option('--file-name', '-f', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='read chunk from TIFF file.')
@click.option('--voxel-offset', '-v', type=click.INT, nargs=3, callback=default_none,
              help='global offset of this chunk')
@click.option('--voxel-size', '-s', type=click.INT, nargs=3, default=None, callback=default_none,
              help='physical size of voxels. The unit is assumed to be nm.')
@click.option('--dtype', '-d',
              type=click.Choice(['uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64', 'float16']),
              help='convert to data type')
@click.option('--output-chunk-name', '-o', type=str, default='chunk',
              help='chunk name in the global state')
@operator
def read_tif(tasks, name: str, file_name: str, voxel_offset: tuple,
             voxel_size: tuple, dtype: str, output_chunk_name: str):
    """Read tiff files."""
    for task in tasks:
        if task is not None:
            start = time()
            task[output_chunk_name] = Chunk.from_tif(
                file_name,
                dtype=dtype,
                voxel_offset=voxel_offset,
                voxel_size=voxel_size)
            task['log']['timer'][name] = time() - start
        yield task


@main.command('save-tif')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--file-name', '-f', default=None,
    type=click.Path(dir_okay=False, resolve_path=True), 
    help='file name of tif file, the extention should be .tif or .tiff')
@click.option('--dtype', '-t', type=click.Choice(['uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64']),
    default=None, help='convert to this data type.')
@click.option('--compression', '-c', 
    type=click.Choice(['', 'zlib', 'lzw', 'lzma', 'delta', 'packints', 'jpeg']),
    default = 'zlib', help = 'encoders that supported by tifffile' 
)
@operator
def save_tif(tasks, input_chunk_name: str, file_name: str, dtype: str, compression: str):
    """Save chunk as a TIF file."""
    for task in tasks:
        if task is not None:
            chunk = task[input_chunk_name]
            chunk = chunk.astype(dtype)
            chunk.to_tif(file_name, compression=compression)
        yield task


@main.command('load-h5')
@click.option('--name', type=str, default='load-h5',
              help='read file from local disk.')
@click.option('--file-name', '-f', type=str, required=True,
              help='read chunk from file, support .h5')
@click.option('--dataset-path', '-d', type=str, default=None,
              help='the dataset path inside HDF5 file.')
@click.option('--dtype', '-e',
              type=click.Choice(['float32', 'float64', 'uint16', 'uint32', 'uint64', 'uint8']),
              default=None, help='transform data type.')
@click.option('--voxel-offset', '-v', type=click.INT, nargs=3, default=None,
              callback=default_none, help='voxel offset of the dataset in hdf5 file.')
@click.option('--voxel-size', '-x', type=click.INT, nargs=3, 
    default=None, callback=default_none, 
    help='physical size of voxels. The unit is assumed to be nm.')
@click.option('--cutout-start', '-t', type=click.INT, nargs=3, callback=default_none,
              help='cutout voxel offset in the array')
@click.option('--cutout-stop', '-p', type=click.INT, nargs=3, callback=default_none,
               help='cutout stop corrdinate.')
@click.option('--cutout-size', '-s', type=click.INT, nargs=3, callback=default_none,
               help='cutout size of the chunk.')
@click.option('--set-bbox/--no-set-bbox', default=False, 
    help='set up bounding box in the task or not')
@click.option('--output-chunk-name', '-o',
              type=str, default=DEFAULT_CHUNK_NAME,
              help='chunk name in the global state')
@operator
def load_h5(tasks, name: str, file_name: str, dataset_path: str,
            dtype: str, voxel_offset: tuple, voxel_size: tuple, 
            cutout_start: tuple, cutout_stop: tuple, 
            cutout_size: tuple, set_bbox: bool,
            output_chunk_name: str):
    """Read HDF5 files."""
    for task in tasks:
        if task is not None:
            start = time()
            if 'bbox' in task and cutout_start is None:
                bbox = task['bbox']
                logging.info(f'bbox: {bbox}')
                cutout_start_tmp = bbox.minpt
                cutout_stop_tmp = bbox.maxpt
                cutout_size_tmp = cutout_stop_tmp - cutout_start_tmp

                if not file_name.endswith('.h5'):
                    file_name = f'{file_name}{bbox.string}.h5'
            else:
                cutout_start_tmp = cutout_start
                cutout_stop_tmp = cutout_stop
                cutout_size_tmp = cutout_size

            chunk = Chunk.from_h5(
                file_name,
                dataset_path=dataset_path,
                voxel_offset=voxel_offset,
                voxel_size=voxel_size,
                cutout_start=cutout_start_tmp,
                cutout_size=cutout_size_tmp,
                cutout_stop=cutout_stop_tmp,
                dtype=dtype,
            )
            if chunk is not None and dtype is not None:
                chunk = chunk.astype(dtype)
            task[output_chunk_name] = chunk
            # make a bounding box for others operators to follow
            if set_bbox and chunk is not None:
                task['bbox'] = chunk.bbox

            task['log']['timer'][name] = time() - start
        yield task


@main.command('save-h5')
@click.option('--input-name', '-i',
              type=str, default='chunk', help='input chunk name')
@click.option('--file-name', '-f',
              type=click.Path(dir_okay=True, resolve_path=False), required=True,
              help='file name or prefix of output HDF5 file.')
@click.option('--chunk-size', '-s', type=click.INT, nargs=3,
              default=None, callback=default_none,
              help='save the big volume as chunks.')
@click.option('--compression', '-c', type=click.Choice(["gzip", "lzf", "szip"]),
              default="gzip", help="compression used in the dataset.")
@click.option('--with-offset/--without-offset', default=True, type=bool,
              help='add voxel_offset dataset or not.')
@click.option('--voxel-size', '-v',
    default=None, type=click.INT, callback=default_none, nargs=3,
    help='voxel size of this chunk.'
)
@click.option('--dtype', '-d', default=None, type=str, 
    help='data type conversion.')
@click.option('--touch/--no-touch', default=True, 
help = 'create an empty file if the input is None.'
)
@operator
def save_h5(tasks, input_name: str, file_name: str, chunk_size: tuple, 
        compression: str, with_offset: bool, voxel_size: tuple, dtype: str, touch: bool):
    """Save chunk to HDF5 file."""
    for task in tasks:
        if task is not None:
            data = task[input_name]
            if isinstance(data, Chunk):
                if not file_name.endswith('.h5'):
                    file_name = f'{file_name}{data.bbox.string}.h5'

                if dtype is not None:
                    data = data.astype(dtype)
                data.to_h5(
                    file_name, with_offset, 
                    chunk_size=chunk_size, 
                    compression=compression,
                    voxel_size=voxel_size)
            elif isinstance(data, Synapses):
                data.to_h5(file_name)
            elif data is None:
                if touch:
                    if not file_name.endswith('.h5'):
                        bbox = task['bbox']
                        file_name = f'{file_name}{bbox.string}.h5'
                    Path(file_name).touch()
            else:
                raise ValueError(f'unsuported type of input data: {data}')
        yield task


@main.command('save-pngs')
@click.option('--name', type=str, default='save-pngs', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--dtype', '-t', type=click.Choice(['uint8', 'uint16']), 
    default='uint8', help='data type. only support uint8 and uint16')
@click.option('--output-path', '-o',
              type=str, default='./pngs/', help='output path of saved 2d images formated as png.')
@operator
def save_pngs(tasks, name, input_chunk_name, dtype, output_path):
    """Save as 2D PNG images."""
    operator = SavePNGsOperator(
        output_path=output_path, 
        dtype=dtype)

    for task in tasks:
        if task is not None:
            operator(task[input_chunk_name])
        yield task


@main.command('skeletonize')
@click.option('--name', '-n', type=str, default='skeletonize',
              help='create centerlines of objects in a segmentation chunk.')
@click.option('--input-chunk-name', '-i', type=str, default=DEFAULT_CHUNK_NAME,
              help='input chunk name.')
@click.option('--output-name', '-o', type=str, default='skeletons')
@click.option('--voxel-size', type=click.INT, nargs=3, required=True,
              help='voxel size of segmentation chunk (zyx order)')
@click.option('--output-path', type=str, required=True,
              help='output path with protocols, such as file:///bucket/my/path')
@operator
def skeletonize(tasks, name, input_chunk_name, output_name, voxel_size, output_path):
    """Skeletonize the neurons/objects in a segmentation chunk"""
    operator = SkeletonizeOperator(output_path, name=name)
    for task in tasks:
        if task is not None:
            seg = task[input_chunk_name]
            skels = operator(seg, voxel_size)
            task[output_name] = skels
        yield task


@main.command('delete-task-in-queue')
@click.option('--name', type=str, default='delete-task-in-queue',
              help='name of this operator')
@operator
def delete_task_in_queue(tasks, name):
    """Delete the task in queue."""
    for task in tasks:
        if task is not None:
            if state['dry_run']:
                logging.info('skip deleting task in queue!')
            else:
                queue = task['queue']
                task_handle = task['task_handle']
                queue.delete(task_handle)
                logging.info(f'deleted task {task_handle} in queue: {queue.queue_name}')
        yield task


@main.command('delete-var')
@click.option('--var-names', '-v',
              type=str, required=True, help='the variable names to be deleted')
@operator
def delete_var(tasks, var_names: str):
    """Delete a Chunk in task to release RAM"""
    for task in tasks:
        if task is not None:
            logging.info(f'delete data: {var_names}')
            for var_name in var_names.split(','):
                del task[var_name]
        yield task
 

@main.command('load-precomputed')
@click.option('--name',
              type=str, default='load-precomputed', help='name of this operator')
@click.option('--volume-path', '-v',
              type=str, required=True, help='volume path')
@click.option('--mip', '-m',
              type=click.INT, default=None, help='mip level of the cutout.')
@click.option('--expand-margin-size', '-e',
              type=click.INT, nargs=3, default=(0, 0, 0),
              help='include surrounding regions of output bounding box.')
@click.option('--expand-direction', '-d',
    type=click.Choice(['-1', '1'],), default=None,
    help="""if it is -1, only expand at -z,-y,-x direction. 
    if it is None[default], expand at both directions.""")
@click.option('--chunk-start', '-s',
              type=click.INT, nargs=3, default=None, callback=default_none,
              help='chunk offset in volume.')
@click.option('--chunk-size', '-z',
              type=click.INT, nargs=3, default=None, callback=default_none,
              help='cutout chunk size.')
@click.option('--fill-missing/--no-fill-missing',
              default=True, help='fill the missing chunks in input volume with zeros ' +
              'or not, default is false')
@click.option('--validate-mip', 
              type=click.INT, default=None, help='validate chunk using higher mip level')
@click.option('--blackout-sections/--no-blackout-sections',
    default=False, help='blackout some sections. ' +
    'the section ids json file should named blackout_section_ids.json. default is False.')
@click.option('--use-https/--use-credential', default=False,
    help='if we read from a public dataset in cloud storage, it is required to use https.')
@click.option(
    '--output-chunk-name', '-o',
    type=str, default=DEFAULT_CHUNK_NAME, 
    help='Variable name to store the cutout to for later retrieval.'
    + 'Chunkflow operators by default operates on a variable named "chunk" but' +
    ' sometimes you may need to have a secondary volume to work on.'
)
@operator
def load_precomputed(tasks, name: str, volume_path: str, mip: int,
        chunk_start: tuple, chunk_size: tuple,
        expand_margin_size: tuple, expand_direction: str,
        fill_missing: bool, validate_mip: int, blackout_sections: bool,
        use_https: bool, output_chunk_name: str):
    """Cutout chunk from volume."""
    if mip is None:
        mip = state['mip']
    assert mip >= 0

    if expand_direction is not None:
        # only -1 or 1
        expand_direction = int(expand_direction)
    
    operator = LoadPrecomputedOperator(
        volume_path,
        mip=mip,
        expand_margin_size=expand_margin_size,
        expand_direction=expand_direction,
        fill_missing=fill_missing,
        validate_mip=validate_mip,
        blackout_sections=blackout_sections,
        use_https=use_https,
        dry_run=state['dry_run'],
        name=name)

    for task in tasks:
        if task is not None:
            if 'bbox' in task:
                bbox = task['bbox']
            else:
                # use bounding box of volume
                if chunk_start is None:
                    chunk_start = operator.vol.mip_bounds(mip).minpt[::-1]
                else:
                    chunk_start = Vec(*chunk_start)

                if chunk_size is None:
                    chunk_stop = operator.vol.mip_bounds(mip).maxpt[::-1]
                    chunk_size = chunk_stop - chunk_start
                else:
                    chunk_size = Vec(*chunk_size)
                bbox = BoundingBox.from_delta(chunk_start, chunk_size)

            start = time()
            # assert output_chunk_name not in task
            task[output_chunk_name] = operator(bbox)
            task['log']['timer'][name] = time() - start
            task['cutout_volume_path'] = volume_path
        yield task


@main.command('load-zarr')
@click.option('--store', '-f', type=str, required=True,
    help='Zarr store path')
@click.option('--path', '-p', type=str, default = None,
    help = 'Zarr path in the store')
@click.option('--chunk-start', '-s', type=click.INT, nargs=3, default=None,
    help='voxel offset or start')
@click.option('--chunk-size', '-z', type=click.INT, nargs=3, default=None,
    help='chunk size')
@click.option('--voxel-size', '-v', type=click.FLOAT, nargs=3, default=None)
@click.option('--backend', '-b', type=str, default='NestedDirectoryStore',
    help='the storage backend.')
@click.option('--output-chunk-name', '-o', type=str, default=DEFAULT_CHUNK_NAME,
    help='output chunk name.')
@operator
def load_zarr(tasks, store: str, path: str, chunk_start: tuple, voxel_size: tuple, 
        chunk_size: tuple, backend: str, output_chunk_name: str):
    """Load Zarr arrays."""
    if backend == 'NestedDirectoryStore':
        store = zarr.NestedDirectoryStore(store)
    z = zarr.open(store, mode='r', path=path)
    attrs = z.attrs.asdict()
    
    # Note that this is the physical
    if 'offset' in attrs:
        physical_volume_offset = Cartesian.from_collection(attrs['offset'])
        volume_offset = physical_volume_offset / voxel_size
    elif 'voxel_offset' in attrs:
        volume_offset = Cartesian.from_collection(attrs['voxel_offset'])
    else:
        print('no voxel offset, set default value: 0x0x0')
        volume_offset = Cartesian(0, 0, 0)

    if voxel_size is None:
        if 'resolution' in attrs:
            voxel_size = Cartesian.from_collection(attrs['resolution'])
        elif 'voxel_size' in attrs:
            voxel_size = Cartesian.from_collection(attrs['voxel_size'])
        else:
            print('no voxel size, set default value: 1x1x1')
            voxel_size = Cartesian(1, 1, 1)
            # raise ValueError(f'no voxel size attribute!')
    for task in tasks:
        if task is not None:
            if chunk_size is None and chunk_start is None and 'bbox' not in task:
                arr = z[:]
                voxel_offset = volume_offset
            else:
                if chunk_start is not None and chunk_size is not None:
                    bbox = BoundingBox.from_delta(chunk_start, chunk_size)
                elif 'bbox' in task:
                    bbox = task['bbox']
                    chunk_start = bbox.start
                    chunk_size = bbox.shape
                else:
                    raise ValueError(f'bounding box not defined.')
                arr_start = bbox.start - volume_offset
                arr_bbox = BoundingBox.from_delta(arr_start, bbox.shape)
                arr = z[arr_bbox.slices]
                voxel_offset = chunk_start
            chunk = Chunk(arr, voxel_offset=voxel_offset, voxel_size=voxel_size) 
            task[output_chunk_name] = chunk
        yield task


@main.command('save-zarr')
@click.option('--store', '-s', type=str, required=True,
    help = 'Zarr store path')
@click.option('--shape', '-s', type=click.INT, nargs=3,
    default=None, callback=default_none,
    help='shape of the whole volume.')
@click.option('--input-chunk-name', '-i', type=str, default=DEFAULT_CHUNK_NAME,
    help='input chunk name.')
@operator
def save_zarr(tasks, store: str, shape: tuple, input_chunk_name: str):
    """Load Zarr arrays."""
    
    if os.path.exists(store):
        zarr_store = zarr.open(store, mode='w')
    else:
        assert shape is not None
        zarr_store = zarr.open(store, mode='w', shape=shape,)
    for task in tasks:
        if task is not None:
            chunk = task[input_chunk_name]
            if not os.path.exists(store):
                # create it and store the whole array here.
                za[:] = chunk.array
            else:
                if chunk.ndim == 4: 
                    za[(slice(None),) + chunk.slices] = chunk.array
                elif chunk.ndim == 3:
                    za[chunk.slices] = chunk.array
                else:
                    raise ValueError(f'only support 3D and 4D array for now, but get {chunk.ndim}')
        yield task


@main.command('remap-segmentation')
@click.option('--input-chunk-name', '-i',
    type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name.')
@click.option('--output-chunk-name', '-o',
    type=str, default=DEFAULT_CHUNK_NAME, help='output chunk name.')
@operator
def remap_segmentation(tasks, input_chunk_name, output_chunk_name):
    """Renumber a serials of chunks."""
    # state['remap_start_id'] = 0
    start_id = 0
    for task in tasks:
        if task is not None:
            seg = task[input_chunk_name]
            assert seg.is_segmentation
            if not isinstance(seg, Segmentation):
                seg = Segmentation.from_chunk(seg)

            seg, start_id = seg.remap(start_id)
            task[output_chunk_name] = seg
        yield task


@main.command('evaluate-segmentation')
@click.option("--segmentation-chunk-name",
              "-s",
              type=str,
              default="chunk",
              help="chunk name of segmentation")
@click.option("--groundtruth-chunk-name",
              "-g",
              type=str,
              default="groundtruth")
@click.option('--output', '-o',
    type=str, default='seg_score',
    help='segmentation evaluation result name.')
@operator
def evaluate_segmenation(tasks, segmentation_chunk_name,
                         groundtruth_chunk_name, output):
    """Evaluate segmentation by split/merge error.
    """
    for task in tasks:
        if task is not None:
            seg = Segmentation(task[segmentation_chunk_name])
            groundtruth = Segmentation(task[groundtruth_chunk_name])
            task[output] = seg.evaluate(groundtruth)
        yield task


@main.command('downsample')
@click.option('--input-chunk-name', '-i', type=str, default=DEFAULT_CHUNK_NAME,
    help = 'input chunk name')
@click.option('--output-chunk-name', '-o', type=str, default=DEFAULT_CHUNK_NAME,
    help='output chunk name')
@click.option('--factor', '-f', type=click.INT, nargs=3, default=(2,2,2),
    help='downsample factor in zyx. The default is 2x2x2.')
@operator
def downsample(tasks, input_chunk_name: str, output_chunk_name: str, factor: tuple):
    for task in tasks:
        if task is not None:
            chunk = task[input_chunk_name]
            if chunk.is_image:
                arr = tinybrain.downsample_with_averaging(chunk.array, factor)[0]
            elif chunk.is_segmentation:
                arr = tinybrain.downsample_segmentation(chunk.array, factor)[0]
            else:
                raise TypeError(f'only support image or segmentation, but got: {chunk.dtype}')
                
            factor = Cartesian.from_collection(factor)
            voxel_offset = chunk.voxel_offset // factor
            voxel_size = chunk.voxel_size * factor

            output_chunk = Chunk(arr, 
                voxel_offset=voxel_offset,
                voxel_size=voxel_size,
                layer_type=chunk.layer_type)
            task[output_chunk_name] = output_chunk
        yield task

@main.command('downsample-upload')
@click.option('--name',
              type=str, default='downsample-upload', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default='chunk', help='input chunk name')
@click.option('--volume-path', '-v', type=str, help='path of output volume')
@click.option('--factor', '-f', type=click.INT, nargs=3, default=(2, 2, 2), 
    help='downsampling factor in z,y,x.')
@click.option('--chunk-mip', '-c', type=click.INT, default=None, help='input chunk mip level')
@click.option('--start-mip', '-s', 
    type=click.INT, default=None, help='the start uploading mip level.')
@click.option('--stop-mip', '-p',
    type=click.INT, default=5, help='stop mip level. the indexing follows python style and ' +
    'the last index is exclusive.')
@click.option('--fill-missing/--no-fill-missing',
              default=True, help='fill missing or not when there is all zero blocks.')
@operator
def downsample_upload(tasks, name, input_chunk_name, volume_path, 
                      factor, chunk_mip, start_mip, stop_mip, fill_missing):
    """Downsample chunk and upload to volume."""
    if chunk_mip is None:
        chunk_mip = state['mip']

    operator = DownsampleUploadOperator(
        volume_path,
        factor=factor,
        chunk_mip=chunk_mip,
        start_mip=start_mip,
        stop_mip=stop_mip,
        fill_missing=fill_missing,
        name=name)

    for task in tasks:
        if task is not None:
            start = time()
            operator(task[input_chunk_name])
            task['log']['timer'][name] = time() - start
        yield task


@main.command('gaussian-filter')
@click.option('--name', type=str, default='gaussian-filter', help='name of operator')
@click.option('--input-chunk-name', '-i',
    type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--sigma', '-s',
    type=click.INT, default=1, help='standard deviation of gaussian kernel')
@operator
def gaussian_filter(tasks, name, input_chunk_name, sigma):
    """2D Gaussian blurring operated in-place."""
    for task in tasks:
        if task is not None:
            start = time()
            chunk = task[input_chunk_name]
            chunk.gaussian_filter_2d(sigma)
            task['log']['timer'][name] = time() - start
        yield task


@main.command('log-summary')
@click.option('--log-dir', '-l',
              type=click.Path(exists=True, dir_okay=True, readable=True),
              default='./log', help='directory of json log files.')
@click.option('--output-size', '-s', 
    type=click.INT, nargs=3, default=None, callback=default_none,
    help='output size for each task. will be used for computing speed.')
@generator
def log_summary(log_dir, output_size):
    """Compute the statistics of large scale run."""
    df = load_log(log_dir)
    print_log_statistics(df, output_size=output_size)

    task = get_initial_task()
    yield task
        

@main.command('normalize-intensity')
@click.option('--name', type=str, default='normalize-intensity', help='name of operator')
@click.option('--input-chunk-name', '-i', type=str, 
    default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--output-chunk-name', '-o', type=str,
    default=DEFAULT_CHUNK_NAME, help='output chunk name')
@operator
def normalize_intensity(tasks, name, input_chunk_name, output_chunk_name):
    """transform gray image to float (-1:1). x=(x-127.5) - 1.0"""
    for task in tasks:
        if task is not None:
            start = time()
            chunk = task[input_chunk_name]
            assert np.issubdtype(chunk.dtype, np.uint8)
            chunk = chunk.astype('float32')
            chunk /= 127.5
            chunk -= 1.0
            task[output_chunk_name] = chunk
            task['log']['timer'][name] = time() - start
        yield task


@main.command('normalize-contrast')
@click.option('--name', type=str, default='normalize-contrast-nkem',
              help='name of operator.')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--output-chunk-name', '-o',
              type=str, default=DEFAULT_CHUNK_NAME, help='output chunk name')
@click.option('--lower-clip-fraction', '-l', type=click.FLOAT, default=0.01, 
              help='lower intensity fraction to clip out.')
@click.option('--upper-clip-fraction', '-u', type=click.FLOAT, default=0.01, 
              help='upper intensity fraction to clip out.')
@click.option('--minval', type=click.INT, default=1, 
              help='the minimum intensity of transformed chunk.')
@click.option('--maxval', type=click.INT, default=255,
              help='the maximum intensity of transformed chunk.')
@click.option('--per-section/--whole', default=True, 
help='per section normalization or normalize the whole chunk.')
@operator
def normalize_contrast(tasks, 
        name: str, input_chunk_name: str, output_chunk_name: str, 
        lower_clip_fraction: float, upper_clip_fraction: float, 
        minval: int, maxval: int, per_section: bool):
    """Normalize the section contrast using precomputed histograms."""
    
    for task in tasks:
        if task is not None:
            start = time()
            chunk = task[input_chunk_name]
            chunk = chunk.clone()
            chunk = Image.from_chunk(chunk)
            chunk.normalize_contrast(
                lower_clip_fraction=lower_clip_fraction,
                upper_clip_fraction=upper_clip_fraction,
                minval=minval,
                maxval=maxval,
                per_section=per_section) 
            task[output_chunk_name] = chunk
            task['log']['timer'][name] = time() - start
        yield task


@main.command('normalize-section-shang')
@click.option('--name',
              type=str,
              default='normalize-section-mu',
              help='name of operator.')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--output-chunk-name', '-o',
              type=str, default=DEFAULT_CHUNK_NAME, help='output chunk name')
@click.option('--nominalmin',
              type=click.FLOAT,
              default=None,
              help='targeted minimum of transformed chunk.')
@click.option('--nominalmax',
              type=click.FLOAT,
              default=None,
              help='targeted maximum of transformed chunk.')
@click.option('--clipvalues',
              type=bool,
              default=False,
              help='clip transformed values to be within the target range.')
@operator
def normalize_section_shang(tasks, name, input_chunk_name, output_chunk_name, 
                            nominalmin, nominalmax, clipvalues):
    """Normalize voxel values based on slice min/max within the chunk, Shang's method.
    The transformed chunk has floating point values.
    """

    for task in tasks:
        if task is not None:
            start = time()
            chunk = task[input_chunk_name]
            chunk = chunk.normalize_section_shang(nominalmin, nominalmax, clipvalues)
            task[output_chunk_name] = chunk
            task['log']['timer'][name] = time() - start
        yield task


@main.command('plugin')
@click.option('--name',
              type=str,
              default='plugin-1',
              help='name of plugin. Multiple plugins should have different names.')
@click.option('--input-names', '-i',
              type=str, default=None, help='input names with delimiter of comma')
@click.option('--output-names', '-o',
              type=str, default=None, help='output names with dilimiter of comma')
@click.option('--file', '-f', type=str, help='''python file to call. 
                If it is just a name rather than full path, 
                we\'ll look for it in the plugin folder.''')
@click.option('--args', '-a',
              type=str, default=None,
              help='arguments of plugin. keywords should be like: var1=3;var2=(1,2);var3=0.4')
@operator
def plugin(tasks, name: str, input_names: str, output_names: str, file: str, args: str):
    """Insert custom program as a plugin.
    The custom python file should contain a callable named "exec" such that 
    a call of `exec(chunk, args)` can be made to operate on the chunk.
    """
    operator = Plugin(file, name=name)

    for task in tasks:
        if task is not None:
            start = time()
            if input_names is not None:
                input_name_list = input_names.split(',')
                inputs = []
                for input_name in input_name_list:
                    if input_name == 'None':
                        inputs.append(None)
                    else:
                        inputs.append(task[input_name])
            else:
                inputs = []
            outputs = operator(inputs, args=args)
            if isinstance(outputs, list) or isinstance(outputs, tuple): 
                output_name_list = output_names.split(',')
                assert len(outputs) == len(output_name_list)
                for output_name, output in zip(output_name_list, outputs):
                    task[output_name] = output
            elif output_names is not None:
                assert ',' not in output_names
                task[output_names] = outputs
            else:
                assert outputs is None

            task['log']['timer'][name] = time() - start
        yield task


@main.command('connected-components')
@click.option('--name', type=str, default='connected-components', 
              help='threshold a map and get the targets.')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, 
              help='input chunk name')
@click.option('--output-chunk-name', '-o',
              type=str, default=DEFAULT_CHUNK_NAME, 
              help='output chunk name')
@click.option('--threshold', '-t', type=click.FLOAT, default=None,
              help='threshold to cut the map.')
@click.option('--connectivity', '-c', 
              type=click.Choice(['6', '18', '26']),
              default='6', help='number of neighboring voxels used. Default is 6.')
@operator
def connected_components(tasks, name: str, input_chunk_name: str, output_chunk_name: str, 
                         threshold: float, connectivity: str):
    """Threshold the probability map to get a segmentation."""
    connectivity = int(connectivity)
    for task in tasks:
        if task is not None:
            start = time()
            task[output_chunk_name] = task[input_chunk_name].connected_component(
                threshold=threshold, connectivity=connectivity)
            task['log']['timer'][name] = time() - start
        yield task


@main.command('copy-var')
@click.option('--from-name', '-f',
              type=str,
              default='chunk',
              help='Variable to be copied')
@click.option('--to-name', '-t', type=str, default='chunk', help='New variable name')
@click.option('--deep-copy/--shallow-copy', type=bool, default=True,
    help='really copy data or just create a new name or reference.')
@operator
def copy_var(tasks, from_name: str, to_name: str, deep_copy: bool):
    """Deep or shallow copy a variable."""
    for task in tasks:
        if task is not None:
            if deep_copy:
                task[to_name] = deepcopy(task[from_name])
            else:
                task[to_name] = task[from_name]
        yield task


@main.command('inference')
@click.option('--name', type=str, default='inference', 
              help='name of this operator')
@click.option('--convnet-model', '-m',
              type=str, default=None, help='convnet model path or type.')
@click.option('--convnet-weight-path', '-w',
              type=str, default=None, help='convnet weight path')
@click.option('--input-patch-size', '-s',
              type=click.INT, nargs=3, required=True, help='input patch size')
@click.option('--output-patch-size', '-z', type=click.INT, nargs=3, default=None, 
              callback=default_none, help='output patch size')
@click.option('--output-patch-overlap', '-v', type=click.INT, nargs=3, 
              default=(4, 64, 64), help='patch overlap')
@click.option('--output-crop-margin', type=click.INT, nargs=3,
              default=None, callback=default_none, help='margin size of output chunk cropping.')
@click.option('--patch-num', '-n', default=None, callback=default_none,
              type=click.INT, nargs=3, help='patch number in z,y,x.')
@click.option('--num-output-channels', '-c',
              type=click.INT, default=3, help='number of output channels')
@click.option('--dtype', '-d', type=click.Choice(['float32', 'float16']),
              default='float32', help="""Even if we perform inference using float16, 
                    the result will still be converted to float32.""")
@click.option('--framework', '-f',
              type=click.Choice(['universal', 'identity', 'pytorch']),
              default='universal', help='inference framework')
@click.option('--batch-size', '-b',
              type=click.INT, default=1, help='mini batch size of input patch.')
@click.option('--bump', type=click.Choice(['wu', 'zung']), default='wu',
              help='bump function type (only support wu now!).')
@click.option('--mask-output-chunk/--no-mask-output-chunk', default=False,
              help='mask output chunk will make the whole chunk like one output patch. '
              + 'This will also work with non-aligned chunk size.')
@click.option('--mask-myelin-threshold', '-y', default=None, type=click.FLOAT,
              help='mask myelin if netoutput have myelin channel.')
@click.option('--input-chunk-name', '-i',
              type=str, default='chunk', help='input chunk name')
@click.option('--output-chunk-name', '-o',
              type=str, default='chunk', help='output chunk name')
@operator
def inference(tasks, name, convnet_model, convnet_weight_path, input_patch_size,
              output_patch_size, output_patch_overlap, output_crop_margin, patch_num,
              num_output_channels, dtype, framework, batch_size, bump, mask_output_chunk,
              mask_myelin_threshold, input_chunk_name, output_chunk_name):
    """Perform convolutional network inference for chunks."""
    with Inferencer(
        convnet_model,
        convnet_weight_path,
        input_patch_size=input_patch_size,
        output_patch_size=output_patch_size,
        num_output_channels=num_output_channels,
        output_patch_overlap=output_patch_overlap,
        output_crop_margin=output_crop_margin,
        patch_num=patch_num,
        framework=framework,
        dtype=dtype,
        batch_size=batch_size,
        bump=bump,
        mask_output_chunk=mask_output_chunk,
        mask_myelin_threshold=mask_myelin_threshold,
        dry_run=state['dry_run']) as inferencer:
        
        for task in tasks:
            if task is not None:
                if 'log' not in task:
                    task['log'] = {'timer': {}}
                start = time()

                task[output_chunk_name] = inferencer(
                    task[input_chunk_name])

                task['log']['timer'][name] = time() - start
                task['log']['compute_device'] = inferencer.compute_device
            yield task


@main.command('multiply')
@click.option('--input-names', '-i', type=str, default=DEFAULT_CHUNK_NAME)
@click.option('--output-names', '-o', type=str, default=None)
@click.option('--multiplier-name', '-m', type=str, required=True,
    help='multiplier chunk name.')
@operator
def multiply(tasks, input_names: str, output_names: str, multiplier_name: str):
    """Multiply chunks with another chunk"""
    input_names = input_names.split(',')
    if output_names is None:
        output_names = input_names
    else:
        output_names = output_names.split(',')
        assert len(input_names)==len(output_names), \
            'the number of input and output chunks should be the same'

    for task in tasks:
        if task is not None:
            for input_name, output_name in zip(input_names, output_names):
                task[output_name] = task[input_name] * task[multiplier_name]
        
        yield task


@main.command('mask')
@click.option('--name', type=str, default='mask', help='name of this operator')
@click.option('--input-names', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk names')
@click.option('--output-names', '-o',
              type=str, default=None, help='output chunk names')
@click.option('--volume-path', '-v',
              type=str, required=True, help='mask volume path')
@click.option('--mip', '-m', 
              type=click.INT, default=5, help='mip level of mask')
@click.option('--inverse/--no-inverse',
              default=False,
              help='inverse the mask or not. default is True. ' +
              'the mask will be multiplied to chunk.')
@click.option('--fill-missing/--no-fill-missing',
              default=True,
              help='fill missing blocks with black or not. ' +
              'default is False.')
@operator
def mask(tasks, name, input_names: str, output_names: str, volume_path: str, 
         mip: int, inverse: bool, fill_missing: bool):
    """Mask the chunk. The mask could be in higher mip level and we
    will automatically upsample it to the same mip level with chunk.
    """
    if output_names is None:
        output_names = input_names
        
    operator = MaskOperator(volume_path,
                            mip,
                            state['mip'],
                            inverse=inverse,
                            fill_missing=fill_missing,
                            name=name)

    input_names = input_names.split(',')
    output_names = output_names.split(',')
    assert len(input_names) == len(output_names)
    assert len(input_names) > 0
    for task in tasks:
        if task is not None:
            start = time()
            chunks = []
            for input_name in input_names:
                chunks.append(task[input_name])

            operator(chunks)

            for output_name, chunk in zip(output_names, chunks):    
                task[output_name] = chunk
            # Note that mask operation could be used several times,
            # this will only record the last masking operation
            task['log']['timer'][name] = time() - start
        yield task


@main.command('mask-out-objects')
@click.option('--input-chunk-name', '-i', type=str, default=DEFAULT_CHUNK_NAME)
@click.option('--output_chunk_name', '-o', type=str, default=DEFAULT_CHUNK_NAME)
@click.option('--dust-size-threshold', '-d', type=click.INT, default=None,
              help='eliminate small objects with voxel number less than threshold.')
@click.option('--selected-obj-ids', '-s', type=str, default=None,
               help="""a list of segment ids to mesh. This is for sparse meshing. 
               The ids should be separated by comma without space, such as "34,56,78,90"
               it can also be a json file contains a list of ids. The json file path should
               contain protocols, such as "gs://bucket/my/json/file/path.""")
@operator
def mask_out_objects(tasks, input_chunk_name, output_chunk_name,
                     dust_size_threshold: int, selected_obj_ids: List[int]):
    """Mask out objects in a segmentation chunk."""
    if isinstance(selected_obj_ids, str) and selected_obj_ids.endswith('.json'):
        # assume that ids is a json file in the storage path
        json_storage = CloudFiles(os.path.dirname(selected_obj_ids))
        ids_str = json_storage.get_file(os.path.basename(selected_obj_ids))
        selected_obj_ids = set(json.loads(ids_str))
        assert len(selected_obj_ids) > 0
        logging.info(f'number of selected objects: {len(selected_obj_ids)}')

    for task in tasks:
        if task is not None:
            seg = task[input_chunk_name]
            if not isinstance(seg, Segmentation):
                assert isinstance(seg, Chunk)
                seg = Segmentation.from_chunk(seg)

            if dust_size_threshold is not None:
                seg.mask_fragments(dust_size_threshold)
            if selected_obj_ids is not None:
                seg.mask_except(selected_obj_ids)

            task[output_chunk_name] = seg
        yield task


@main.command('crop-margin')
@click.option('--name', type=str, default='crop-margin',
    help='name of this operator')
@click.option('--margin-size', '-m',
    type=click.INT, nargs=6, default=None, callback=default_none,
    help='crop the chunk margin. ' +
            'The default is None and will use the bbox as croping range.')
@click.option('--crop-bbox/--no-crop-bbox', default=False,
    help='adjust the bounding box or not.')
@click.option('--input-chunk-name', '-i',
    type=str, default='chunk', help='input chunk name.')
@click.option('--output-chunk-name', '-o',
    type=str, default='chunk', help='output chunk name.')
@operator
def crop_margin(tasks, name: str, margin_size: tuple, crop_bbox: bool, 
                input_chunk_name: str, output_chunk_name: str):
    """Crop the margin of chunk."""
    for task in tasks:
        if task is not None:
            start = time()
            if margin_size:
                task[output_chunk_name] = task[
                    input_chunk_name].crop_margin(
                    margin_size=margin_size)
                if crop_bbox and 'bbox' in task:
                    bbox = task['bbox']
                    assert isinstance(bbox, BoundingBox)
                    bbox = bbox.adjust(-Cartesian.from_collection(margin_size))
            else:
                # use the output bbox for croping 
                task[output_chunk_name] = task[
                    input_chunk_name].cutout(task['bbox'].slices)
            task['log']['timer'][name] = time() - start
        yield task


@main.command('mesh')
@click.option('--name', type=str, default='mesh', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='name of chunk needs to be meshed.')
@click.option('--mip', '-m',
    type=click.INT, default=None, help='mip level of segmentation chunk.')
@click.option('--voxel-size', '-v', type=click.INT, nargs=3, default=None, callback=default_none, 
    help='voxel size of the segmentation. zyx order.')
@click.option('--output-path', '-o', type=str, default='file:///tmp/mesh/', 
    help='output path of meshes, follow the protocol rule of CloudVolume. \
              The path will be adjusted if there is a info file with precomputed format.')
@click.option('--output-format', '-t', type=click.Choice(['ply', 'obj', 'precomputed']), 
              default='precomputed', help='output format, could be one of ply|obj|precomputed.')
@click.option('--simplification-factor', '-f', type=click.INT, default=100, 
              help='mesh simplification factor.')
@click.option('--max-simplification-error', '-e', type=click.INT, default=40, 
              help='max simplification error.')
@click.option('--skip-ids', '-s', type=str, default=None, 
    help='do not mesh for some specific ids.')
@click.option('--manifest/--no-manifest', default=False, help='create manifest file or not.')
@click.option('--shard/--no-shard', default=False, help='combine meshes as one file')
@operator
def mesh(tasks, name, input_chunk_name, mip, voxel_size, output_path, output_format,
         simplification_factor, max_simplification_error, skip_ids: str, manifest, shard):
    """Perform meshing for segmentation chunk."""
    if mip is None:
        mip = state['mip']

    if skip_ids is not None:
        skip_ids = frozenset(map(int, skip_ids.split(',')))

    operator = MeshOperator(
        output_path,
        output_format,
        mip=mip,
        voxel_size=voxel_size,
        simplification_factor=simplification_factor,
        max_simplification_error=max_simplification_error,
        manifest=manifest,
        skip_ids = skip_ids,
        shard=shard,
    )

    for task in tasks:
        if task is not None:
            start = time()
            operator( task[input_chunk_name] )
            task['log']['timer'][name] = time() - start
        yield task


@main.command('mesh-manifest')
@click.option('--prefix', '-p', type=click.INT, default=None, help='prefix of meshes.')
@click.option('--disbatch/--no-disbatch', default=False, help='use disBatch task index as prefix')
@click.option('--digits', '-d', type=click.INT, default=1, help='number of digits of prefix')
@click.option('--volume-path', '-v', type=str, required=True, help='cloudvolume path of dataset layer.' + 
              ' The mesh directory will be automatically figure out using the info file.')
@generator
def mesh_manifest(prefix: str, 
        disbatch: bool, digits: int, volume_path: str):
    """Generate mesh manifest files."""
    operator = MeshManifestOperator(volume_path)
    if prefix:
        operator(prefix, digits)
    elif disbatch:
            assert 'DISBATCH_REPEAT_INDEX' in os.environ
            prefix = os.environ['DISBATCH_REPEAT_INDEX']
            operator(prefix, digits)
    else:
        for prefix in range(10**digits):
            operator(prefix, digits)
    yield None

@main.command('download-mesh')
@click.option('--volume-path', '-v', type=str, required=True,
    help="volume path of segmentation layer formated as Neuroglancer Precomputed.")
@click.option('--input', '-i', type=str, required=True,
    help='object IDs with comma to separate them. example: 34,25,38. If this is a text file path, we can read the file. It can also be a segmentation chunk name, we can used to get the largest objects ranked by the following parameters: start-rank and stop-rank.')
@click.option('--start-rank', '-s', type=click.INT, default=0, 
    help='starting rank of the object size measured by voxel counts')
@click.option('--stop-rank', '-p', type=click.INT, default=None,
    help='stopping rank of the object size measured by voxel count')
@click.option('--out-pre', '-o', type=str, default='./',
    help='prefix of output file')
@click.option('--out-format', '-f', 
    type=click.Choice(['ply', 'obj'], case_sensitive=True), default='ply',
    help='output format, only support ply and obj for now.')
@operator
def download_mesh(tasks, volume_path: str, input: str, start_rank: int,
        stop_rank: int, out_pre: str, out_format: str):
    vol = CloudVolume(volume_path, green_threads=True)

    for task in tasks:
        if input in task and stop_rank is not None:
            seg = task[input]
            import fastremap
            unique, count = fastremap.unique(seg, return_counts=True)
            # descending order
            unique = unique[1:]
            count = count[1:]

            assert len(count) == len(unique)
            orders = np.argsort(count)[::-1]
            sorted_unique = unique[orders]
            ids = sorted_unique[start_rank:stop_rank]
        else:
            if os.path.isfile(input):
                with open(input, 'r') as file:
                    input = file.read()
            ids = input.replace(' ', '')
            ids = [int(x) for x in ids.split(',')]
        logging.info('downloading meshes...')
        meshes = vol.mesh.get(ids, fuse=False)
        for obj_id, mesh in tqdm(meshes.items(), desc='writing meshes...'):
            fname = f'{out_pre}{obj_id}.{out_format}'
            if out_format == 'ply':
                mesh = mesh.to_ply()
            elif out_format == 'obj':
                mesh = mesh.to_obj()
            else:
                raise ValueError('only support ply and obj for now.')
            with open(fname, 'wb') as f:
                f.write(mesh)

@main.command('neuroglancer')
@click.option('--name', type=str, default='neuroglancer',
              help='name of this operator')
@click.option('--voxel-size', '-v',
              nargs=3, type=click.INT, default=None, callback=default_none,
              help='voxel size of chunk')
@click.option('--port', '-p', type=click.INT, default=None, help='port to use')
@click.option('--inputs', '-i', type=str, default='chunk', 
              help='a list of chunk names separated by comma.')
@operator
def neuroglancer(tasks, name, voxel_size, port, inputs):
    """Visualize the chunk using neuroglancer."""
    operator = NeuroglancerOperator(name=name, port=port, voxel_size=voxel_size)
    for task in tasks:
        if task is not None:
            operator(task, selected=inputs)
        yield task


@main.command('quantize')
@click.option('--input-chunk-name', '-i', type=str, default='chunk', 
    help = 'input chunk name')
@click.option('--output-chunk-name', '-o', type=str, default='chunk', 
    help= 'output chunk name')
@click.option('--mode', type=click.Choice(['xy', 'z']), default='xy',
    help='xy: average of xy channel; z: only the z channel')
@operator
def quantize(tasks, input_chunk_name: str, output_chunk_name: str, mode: str):
    """Transorm the last channel to uint8."""
    for task in tasks:
        if task is not None:
            aff = task[input_chunk_name]
            properties = aff.properties
            aff = AffinityMap(aff)
            aff.set_properties(properties)
            assert isinstance(aff, AffinityMap)
            quantized_image = aff.quantize(mode=mode)
            task[output_chunk_name] = quantized_image
        yield task

@main.command('save-precomputed')
@click.option('--name', type=str, default='save-precomputed', help='name of this operator')
@click.option('--volume-path', '-v', type=str, required=True, help='volume path')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--mip', '-m',
    type=click.INT, default=None, help="mip level to write")
@click.option('--upload-log/--no-upload-log',
              default=False, help='the log will be put inside volume-path')
@click.option('--create-thumbnail/--no-create-thumbnail',
    default=False, help='create thumbnail or not. ' +
    'the thumbnail is a downsampled and quantized version of the chunk.')
@click.option('--intensity-threshold', '-t',
    default=None, type=click.FLOAT,
    help='do not save anything if all voxel intensity is below threshold.'
)
@click.option('--fill-missing/--no-fill', default=False,
    help='save blocks with all zeros or not. Default is not.')
@operator
def save_precomputed(tasks, name: str, volume_path: str, 
        input_chunk_name: str, mip: int, upload_log: bool, 
        create_thumbnail: bool, intensity_threshold: float,
        fill_missing: bool):
    """Save chunk to volume."""
    if mip is None:
        mip = state['mip']

    operator = SavePrecomputedOperator(
        volume_path,
        mip,
        upload_log=upload_log,
        create_thumbnail=create_thumbnail,
        name=name,
        fill_missing=fill_missing,
    )

    for task in tasks:
        if task is not None:
            # the time elapsed was recorded internally
            chunk = task[input_chunk_name]
            if intensity_threshold is not None and np.all(chunk.array < intensity_threshold):
                print(f'average intensity lower than threshold, skip this task.')
                pass
            else:
                operator(chunk, log=task.get('log', {'timer': {}}))
                # task['output_volume_path'] = volume_path

        yield task


@main.command('threshold')
@click.option('--name', type=str, default='threshold', 
              help='threshold a map and get the targets.')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, 
              help='input chunk name')
@click.option('--output-chunk-name', '-o',
              type=str, default=DEFAULT_CHUNK_NAME, 
              help='output chunk name')
@click.option('--threshold', '-t', type=click.FLOAT, default=0.5,
              help='threshold to cut the map.')
@operator 
def threshold(tasks, name, input_chunk_name, output_chunk_name, 
              threshold):
    """Threshold the probability map."""
    for task in tasks:
        if task is not None:
            start = time()
            logging.info('Segment probability map using a threshold...')
            task[output_chunk_name] = task[input_chunk_name].threshold(threshold)
            task['log']['timer'][name] = time() - start
        yield task


@main.command('channel-voting')
@click.option('--input-chunk-name', '-i', type=str, default=DEFAULT_CHUNK_NAME)
@click.option('--output-chunk-name', '-o', type=str, default=DEFAULT_CHUNK_NAME)
@operator
def channel_voting(tasks, input_chunk_name, output_chunk_name):
    """all channels vote to get a uint8 volume. The channel with max intensity wins."""
    for task in tasks:
        task[output_chunk_name] = task[input_chunk_name].channel_voting() 
        yield task


@main.command('view')
@click.option('--name', type=str, default='view', help='name of this operator')
@click.option('--image-chunk-name',
              type=str,
              default='chunk',
              help='image chunk name in the global state')
@click.option('--segmentation-chunk-name',
              type=str,
              default=None,
              help='segmentation chunk name in the global state')
@operator
def view(tasks, name, image_chunk_name, segmentation_chunk_name):
    """Visualize the chunk using cloudvolume view in browser."""
    operator = ViewOperator(name=name)
    for task in tasks:
        if task is not None:
            operator(task[image_chunk_name],
                        seg=segmentation_chunk_name)
        yield task



if __name__ == '__main__':
    main()
