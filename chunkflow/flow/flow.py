#!/usr/bin/env python
import os
from pathlib import Path
from time import time

from typing import Generator, List

from copy import deepcopy

import numpy as np
import click
import json

from numpy.lib.arraysetops import isin

from .lib import *

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec

from chunkflow.lib.aws.sqs_queue import SQSQueue
from chunkflow.lib.bounding_boxes import BoundingBox, BoundingBoxes
from chunkflow.lib.synapses import Synapses

from chunkflow.chunk import Chunk
from chunkflow.chunk.affinity_map import AffinityMap
from chunkflow.chunk.segmentation import Segmentation
from chunkflow.chunk.image.convnet.inferencer import Inferencer

# import operator functions
from .aggregate_skeleton_fragments import AggregateSkeletonFragmentsOperator
from .cloud_watch import CloudWatchOperator
from .read_precomputed import ReadPrecomputedOperator
from .downsample_upload import DownsampleUploadOperator
from .log_summary import load_log, print_log_statistics
from .mask import MaskOperator
from .mesh import MeshOperator
from .mesh_manifest import MeshManifestOperator
from .neuroglancer import NeuroglancerOperator
from .normalize_section_contrast import NormalizeSectionContrastOperator
from .normalize_section_shang import NormalizeSectionShangOperator
from .plugin import Plugin
from .read_pngs import read_png_images
from .write_precomputed import WritePrecomputedOperator
from .write_pngs import WritePNGsOperator
from .setup_env import setup_environment
from .skeletonize import SkeletonizeOperator
from .view import ViewOperator


@main.command('generate-tasks')
@click.option('--layer-path', '-l',
              type=str, default=None,
              help='dataset layer path to fetch dataset information.')
@click.option('--mip', '-m',
              type=int, default=None, help='mip level of the dataset layer.')
@click.option('--roi-start', '-s',
              type=int, default=None, nargs=3, callback=default_none, 
              help='(z y x), start of the chunks')
@click.option('--roi-stop', '-r',
              type=int, nargs=3, default=None, callback=default_none,
              help='stop coordinate of region of interest')
@click.option('--roi-size', '-z',
              type=int, nargs=3, default=None, callback=default_none,
              help='size of region of interest')
@click.option('--chunk-size', '-c',
              type=int, required=True, nargs=3,
              help='(z y x), size/shape of chunks')
@click.option('--grid-size', '-g',
              type=int, default=None, nargs=3, callback=default_none,
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
    type=int, default=None, nargs=3, callback=default_none,
    help='force alignment of block size. Note that the alignment start from (0, 0, 0).')
@click.option('--task-index-start', '-i',
              type=int, default=None, help='starting index of task list.')
@click.option('--task-index-stop', '-p',
              type=int, default=None, help='stop index of task list.')
@click.option('--disbatch/--no-disbatch', '-d',
              default=False, help='use disBatch environment variable or not')
@generator
def generate_tasks(
        layer_path: str, mip: int, roi_start: tuple, roi_stop: tuple,roi_size, chunk_size, 
        grid_size: tuple, file_path: str, queue_name: str, respect_chunk_size: bool,
        aligned_block_size: tuple, task_index_start: tuple, 
        task_index_stop: tuple, disbatch: bool ):
    """Generate a batch of tasks."""
    if mip is None:
        mip = state['mip']
    assert mip >=0 

    """Generate tasks."""
    bboxes = BoundingBoxes.from_manual_setup(
        chunk_size, layer_path=layer_path,
        roi_start=roi_start, roi_stop=roi_stop, 
        roi_size=roi_size, mip=mip, grid_size=grid_size,
        respect_chunk_size=respect_chunk_size,
        aligned_block_size=aligned_block_size
    )
    
    if task_index_start:
        if task_index_stop is None:
            # task_index_stop = task_index_start + 1
            task_index_stop = len(bboxes)
        bboxes = [*bboxes[task_index_start:task_index_stop]]
        logging.info(f'selected task indexes from {task_index_start} to {task_index_stop}')
    elif disbatch:
        assert 'DISBATCH_REPEAT_INDEX' in os.environ
        disbatch_index = int(os.environ['DISBATCH_REPEAT_INDEX'])
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
    print('total number of tasks: ', bbox_num) 

    if queue_name is not None:
        queue = SQSQueue(queue_name)
        queue.send_message_list(bboxes)
    else:
        for bbox_index, bbox in enumerate(bboxes):
            if disbatch:
                assert len(bboxes) == 1
                bbox_index = disbatch_index
            print(f'executing task {bbox_index} in {bbox_num} with bounding box: {bbox.to_filename()}')
            task = get_initial_task()
            task['bbox'] = bbox
            task['bbox_index'] = bbox_index
            task['bbox_num'] = bbox_num
            task['log']['bbox'] = bbox.to_filename()
            yield task


@main.command('skip-task')
@click.option('--pre', '-e', required=True, type=str,
    help='the pre part of result file path')
@click.option('--post', '-t', required=True, type=str,
    help='the post part of result file path. Normally include file extention.')
@click.option('--adjust-size', '-a', default=None, type=int, callback=default_none,
    help='expand or shrink the bounding box. Currently, cloud-volume Bbox only support symetric grow.')
@operator
def skip_task(tasks: Generator, pre: str, post: str, adjust_size: int):
    """if a result file already exists, skip this task."""
    for task in tasks:
        bbox = task['bbox'].clone()
        if adjust_size is not None:
            bbox.grow(adjust_size)
        file_name = pre + bbox.to_filename() + post
        if os.path.exists(file_name):
            print('the result file already exist, skip this task')
            task = None
        yield task


@main.command('skip-all-zero')
@click.option('--input-chunk-name', '-i',
    type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--pre', '-e', type=str, default=None, 
    help = 'pre-path of a file. we would like to keep a trace that this task was executed.')
@click.option('--post', '-t', type=str, default=None,
    help='post-path of a file. normally include the extention of result file.')
@click.option('--adjust-size', '-a', type=int, default=None,
    help='change the bounding box of chunk if it do not match with final result file name.')
@operator
def skip_all_zero(tasks, input_chunk_name: str, pre: str, post: str, adjust_size: int):
    """if chunk has all zero, skip this task."""
    for task in tasks:
        if task is not None:
            chunk = task[input_chunk_name]
            if not np.any(chunk):
                print('all zero chunk, skip this task')
                if pre is not None:
                    bbox = chunk.bbox.clone()
                    bbox.adjust(adjust_size)
                    fname = os.path.join(pre, f'{bbox.to_filename()}{post}')
                    print('create an empty file as mark: ', fname)
                    with open(fname, 'a'):
                        os.utime(fname, None)
                # target task as None and task will be skipped
                task = None
        yield task


@main.command('skip-none')
@click.option('--input-name', '-i',
    type=str, default=DEFAULT_CHUNK_NAME, help='input name')
@operator
def skip_none(tasks, input_name: str):
    """If item is None, skip this task."""
    for task in tasks:
        if task is not None:
            data = task[input_name]
            if data is None:
                # target task as None and task will be skipped
                task = None
        yield task


@main.command('setup-env')
@click.option('--volume-start', required=True, nargs=3, type=int,
              help='start coordinate of output volume in mip 0')
@click.option('--volume-stop', default=None, type=int, nargs=3, callback=default_none,
              help='stop coordinate of output volume (noninclusive like python coordinate) in mip 0.')
@click.option('--volume-size', '-s',
              default=None, type=int, nargs=3, callback=default_none, 
              help='size of output volume.')
@click.option('--layer-path', '-l',
              type=str, required=True, help='the path of output volume.')
@click.option('--max-ram-size', '-r',
              default=15, type=int, help='the maximum ram size (GB) of worker process.')
@click.option('--output-patch-size', '-z',
              type=int, required=True, nargs=3, help='output patch size.')
@click.option('--input-patch-size', '-i',
              type=int, default=None, nargs=3, callback=default_none,
              help='input patch size.')
@click.option('--channel-num', '-c',
              type=int, default=1, 
              help='output patch channel number. It is 3 for affinity map.')
@click.option('--dtype', '-d', type=click.Choice(['uint8', 'float16', 'float32']), 
              default='float32', help='output numerical precision.')
@click.option('--output-patch-overlap', '-o',
              type=int, default=None, nargs=3, callback=default_none,
              help='overlap of patches. default is 50% overlap')
@click.option('--crop-chunk-margin', '-c', 
              type=int, nargs=3, default=None,
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
              type=int, default=3600, help='visibility timeout of the AWS SQS queue.')
@click.option('--thumbnail/--no-thumbnail', default=True, help='create thumbnail or not.')
@click.option('--encoding', '-e',
              type=click.Choice(['raw', 'jpeg', 'compressed_segmentation', 
                                 'fpzip', 'kempressed']), default='raw', 
              help='Neuroglancer precomputed block compression algorithm.')
@click.option('--voxel-size', '-v', type=int, nargs=3, default=(40, 4, 4),
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
            task['log']['bbox'] = bbox.to_filename()
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


@main.command('create-info')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME,
              help="create info for this chunk.")
@click.option('--output-layer-path', '-l', type=str, default="file://.", 
              help='path of output layer.')
@click.option('--channel-num', '-c', type=int, default=1, help='number of channel')
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
@click.option('--voxel-size', '-s', required=True, type=int, nargs=3,
              help='voxel size with unit of nm')
@click.option('--voxel-offset', '-o', default=(0,0,0), type=int, nargs=3,
              help='voxel offset of array')
@click.option('--volume-size', '-v',
              type=int, nargs=3, default=None, callback=default_none,
              help='total size of the volume.')
@click.option('--block-size', '-b',
              type=int, nargs=3, required=True,
              help='chunk size of each file.')
@click.option('--factor', '-f',
              type=int, nargs=3, default=(2,2,2),
              help='hierarchical downsampling factor')
@click.option('--max-mip', '-m',
              type=int, default=0, 
              help = 'maximum mip level.')
@operator
def create_info(tasks,input_chunk_name: str, output_layer_path: str, channel_num: int, 
                layer_type: str, data_type: str, encoding: str, voxel_size: tuple, 
                voxel_offset: tuple, volume_size: tuple, block_size: tuple, factor: tuple, max_mip: int):
    """Create metadata for Neuroglancer Precomputed volume."""
    
    for task in tasks:
        if task is not None:
            if input_chunk_name in task:
                chunk = task[input_chunk_name]
                if chunk.ndim == 3:
                    channel_num = 1
                elif chunk.ndim == 4:
                    channel_num = chunk.shape[0]
                else:
                    raise ValueError('chunk dimension can only be 3 or 4')

                voxel_offset = chunk.voxel_offset
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

            info = CloudVolume.create_new_info(
                channel_num, layer_type=layer_type,
                data_type=data_type,
                encoding=encoding,
                resolution=voxel_size[::-1],
                voxel_offset=voxel_offset[::-1],
                volume_size=volume_size[::-1],
                chunk_size=block_size[::-1],
                factor=Vec(factor),
                max_mip=max_mip)
            vol = CloudVolume(output_layer_path, info=info)
            vol.commit_info()
        yield task


@main.command('fetch-task-from-file')
@click.option('--file-path', '-f',
              type=click.Path(file_okay=True, dir_okay=False, exists=True, 
                              readable=True, resolve_path=True),
              help='file contains bounding boxes or tasks.')
@click.option('--job-index', '-i', 
              type=int, default=None,
              help='index of task in the tasks.')
@click.option('--slurm-job-array/--no-slurm-job-array',
              default=False, help='use the slurm job array '+
              'environment variable to identify task index.')
@click.option('--granularity', '-g',
              type=int, default=1, help='number of tasks to do in one run.')
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
    type=int, default=None, 
    help='visibility timeout of sqs queue; default is using the timeout of the queue.')
@click.option('--num', '-n', type=int, default=-1,
              help='fetch limited number of tasks.' +
              ' This is useful in local cluster to control task time elapse.' + 
              'Negative value will be infinite.')
@click.option('--retry-times', '-r',
              type=int, default=30,
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
        
        print('get task: ', bbox_str)
        bbox = BoundingBox.from_filename(bbox_str)
        
        # record the task handle to delete after the processing
        task = get_initial_task() 
        task['queue'] = queue
        task['task_handle'] = task_handle
        task['bbox'] = bbox
        task['log']['bbox'] = bbox.to_filename()
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
@click.option('--name',
              type=str,
              default='create-chunk',
              help='name of operator')
@click.option('--size', '-s',
              type=int, nargs=3, default=(64, 64, 64), help='the size of created chunk')
@click.option('--dtype',
              type=click.Choice(
                  ['uint8', 'uint32', 'uint16', 'float32', 'float64']),
              default='uint8', help='the data type of chunk')
@click.option('--all-zero/--not-all-zero', default=False, help='all zero or not.')
@click.option('--voxel-offset', '-t',
              type=int, nargs=3, default=(0, 0, 0), help='offset in voxel number.')
@click.option('--voxel-size', '-e',
              type=int, nargs=3, default=(1,1,1), help='voxel size in nm')
@click.option('--output-chunk-name', '-o',
              type=str, default="chunk", help="name of created chunk")
@operator
def create_chunk(tasks, name, size, dtype, all_zero, voxel_offset, voxel_size, output_chunk_name):
    """Create a fake chunk for easy test."""
    print("creating chunk: ", output_chunk_name)
    for task in tasks:
        if task is not None:
            task[output_chunk_name] = Chunk.create(
                size=size, dtype=np.dtype(dtype), 
                all_zero = all_zero,
                voxel_offset=voxel_offset,
                voxel_size=voxel_size)
        yield task


@main.command('load-synapses')
@click.option('--name', '-n', type=str, default='load-synapses', help='name of operator')
@click.option('--file-path', '-f',
    type=click.Path(file_okay=True, dir_okay=True, resolve_path=True),
    required=True, help='files containing synapses. Currently support HDF5 and JSON.')
@click.option('--path-suffix', '-s', type=str, default=None, help='file path suffix.')
@click.option('--c-order/--f-order', default=True,
    help='C order or Fortran order in the file. XYZ is Fortran order, ZYX is C order.')
@click.option('--resolution', '-r', type=int, nargs=3, 
    default=None, callback=default_none, help='resolution of points.')
@click.option('--output-name', '-o', type=str, default='synapses', help='data name of the result.')
@operator
def load_synapses(tasks, name: str, file_path: str, path_suffix: str, c_order: bool, 
        resolution: tuple, output_name: str):
    """Load synapses formated as JSON or HDF5."""
    for task in tasks:
        if task is not None:
            start = time()
            if os.path.isfile(file_path) and (file_path.endswith('.h5') or file_path.endswith('.json')):
                fname = file_path
            elif os.path.isdir(file_path):
                bbox = task['bbox']
                assert path_suffix is not None
                fname = os.path.join(file_path, f'{bbox.to_filename()}{path_suffix}')
            elif not os.path.exists(file_path):
                bbox = task['bbox']
                fname = f'{file_path}{bbox.to_filename()}{path_suffix}'
            else:
                fname = file_path
            assert os.path.isfile(fname), f'can not find file: {fname}'

            if os.path.getsize(fname) == 0:
                task[output_name] = None
            else:
                task[output_name] = Synapses.from_file(
                    fname, 
                    resolution = resolution,
                    c_order = c_order
                )
            
            task['log']['timer'][name] = time() - start
        yield task


@main.command('read-npy')
@click.option('--name', '-n', type=str, default='read-npy', help='name of operator')
@click.option('--file-path', '-f', 
    type=click.Path(file_okay=True, dir_okay=True, resolve_path=True),
    required=True, help='NPY file path')
@click.option('--resolution', '-r', type=float, nargs=3, default=(1,1,1), help='resolution of points.')
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
                    file_path = os.path.join(file_path, f'{bbox.to_filename()}.npy')
                else:
                    file_path = f'{file_path}{bbox.to_filename()}.npy'
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

@main.command('read-json')
@click.option('--name', '-n', type=str, default='read-json', help='name of operator.')
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
                    file_path = os.path.join(file_path, f'{bbox.to_filename()}.json')
                else:
                    file_path = f'{file_path}{bbox.to_filename()}.json'
            assert os.path.exists(file_path)
            with open(file_path, 'r') as file:
                task[output_name] = json.load(file)
            task['log']['timer'][name] = time() - start
        yield task


@main.command('read-nrrd')
@click.option('--name', type=str, default='read-nrrd',
              help='read nrrd file from local disk.')
@click.option('--file-name', '-f', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='read chunk from NRRD file')
@click.option('--voxel-offset', '-v', type=int, nargs=3, default=None, callback=default_none,
              help='global offset of this chunk')
@click.option('--voxel-size', '-s', type=int, nargs=3, default=None, callback=default_none,
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


@main.command('write-nrrd')
@click.option('--name', type=str, default='write-nrrd', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--file-name', '-f', default=None,
    type=click.Path(dir_okay=False, resolve_path=True), 
    help='file name of NRRD file.')
@operator
def write_tif(tasks, name, input_chunk_name, file_name):
    """Write chunk as a NRRD file."""
    for task in tasks:
        if task is not None:
            task[input_chunk_name].to_nrrd(file_name)
        yield task


@main.command('read-pngs')
@click.option('--path-prefix', '-p',
              required=True, type=str,
              help='directory path prefix of png files.')
@click.option('--output-chunk-name', '-o',
              type=str, default=DEFAULT_CHUNK_NAME,
              help='output chunk name')
@click.option('--cutout-offset', '-c',
              type=int, default=(0,0,0), nargs=3,
              help='cutout chunk from an offset')
@click.option('--volume-offset', '-t',
              type=int, nargs=3, default=(0,0,0),
              help = 'the offset of png images volume, could be negative.')
@click.option('--voxel-size', '-x', type=int, nargs=3, default=None, callback=default_none,
              help='physical size of voxels. the unit is assumed to be nm.')
@click.option('--chunk-size', '-s',
              type=int, nargs=3, default=None, callback=default_none,
              help='cutout chunk size')
@operator
def read_pngs(tasks, path_prefix, output_chunk_name, cutout_offset,
                volume_offset, voxel_size, chunk_size):
    """Read a serials of png files."""
    for task in tasks:
        if task is not None:
            if chunk_size is None:
                assert 'bbox' in task, "no chunk_size, we are looking for bounding box in task"
                bbox = task['bbox']
            else:
                bbox = BoundingBox.from_delta(cutout_offset, chunk_size)

            task[output_chunk_name] = read_png_images(
                path_prefix, bbox, 
                volume_offset=volume_offset,
                voxel_size=voxel_size)
        yield task


@main.command('read-tif')
@click.option('--name', type=str, default='read-tif',
              help='read tif file from local disk.')
@click.option('--file-name', '-f', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='read chunk from TIFF file.')
@click.option('--voxel-offset', '-v', type=int, nargs=3, callback=default_none,
              help='global offset of this chunk')
@click.option('--voxel-size', '-s', type=int, nargs=3, default=None, callback=default_none,
              help='physical size of voxels. The unit is assumed to be nm.')
@click.option('--dtype', '-d',
              type=click.Choice(['uint8', 'uint32', 'uint64', 'float32', 'float64', 'float16']),
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


@main.command('write-tif')
@click.option('--name', type=str, default='write-tif', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--file-name', '-f', default=None,
    type=click.Path(dir_okay=False, resolve_path=True), 
    help='file name of tif file, the extention should be .tif or .tiff')
@operator
def write_tif(tasks, name, input_chunk_name, file_name):
    """Write chunk as a TIF file."""
    for task in tasks:
        if task is not None:
            task[input_chunk_name].to_tif(file_name)
        yield task


@main.command('read-h5')
@click.option('--name', type=str, default='read-h5',
              help='read file from local disk.')
@click.option('--file-name', '-f', type=str, required=True,
              help='read chunk from file, support .h5')
@click.option('--dataset-path', '-d', type=str, default=None,
              help='the dataset path inside HDF5 file.')
@click.option('--dtype', '-e',
              type=click.Choice(['float32', 'float64', 'uint32', 'uint64', 'uint8']),
              default=None, help='transform data type.')
@click.option('--voxel-offset', '-v', type=int, nargs=3, default=None,
              callback=default_none, help='voxel offset of the dataset in hdf5 file.')
@click.option('--voxel-size', '-x', type=int, nargs=3, default=None,
              callback=default_none, help='physical size of voxels. The unit is assumed to be nm.')
@click.option('--cutout-start', '-t', type=int, nargs=3, callback=default_none,
              help='cutout voxel offset in the array')
@click.option('--cutout-stop', '-p', type=int, nargs=3, callback=default_none,
               help='cutout stop corrdinate.')
@click.option('--cutout-size', '-s', type=int, nargs=3, callback=default_none,
               help='cutout size of the chunk.')
@click.option('--zero-filling/--no-zero-filling', default=False, type=bool,
    help='if no such file, fill with zero.')
@click.option('--output-chunk-name', '-o',
              type=str, default=DEFAULT_CHUNK_NAME,
              help='chunk name in the global state')
@operator
def read_h5(tasks, name: str, file_name: str, dataset_path: str,
            dtype: str, voxel_offset: tuple, voxel_size: tuple, cutout_start: tuple, 
            cutout_stop: tuple, cutout_size: tuple, zero_filling: bool, output_chunk_name: str):
    """Read HDF5 files."""
    for task in tasks:
        if task is not None:
            start = time()
            if 'bbox' in task and cutout_start is None:
                bbox = task['bbox']
                print('bbox: ', bbox)
                cutout_start_tmp = bbox.minpt
                cutout_stop_tmp = bbox.maxpt
                cutout_size_tmp = cutout_stop_tmp - cutout_start_tmp
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
                zero_filling = zero_filling,
                dtype=dtype,
            )
            if dtype is not None:
                chunk = chunk.astype(dtype)
            task[output_chunk_name] = chunk
            # make a bounding box for others operators to follow
            if 'bbox' not in task:
                task['bbox'] = chunk.bbox

            task['log']['timer'][name] = time() - start
        yield task


@main.command('write-h5')
@click.option('--input-name', '-i',
              type=str, default='chunk', help='input chunk name')
@click.option('--file-name', '-f',
              type=click.Path(dir_okay=True, resolve_path=False), required=True,
              help='file name or prefix of output HDF5 file.')
@click.option('--chunk-size', '-s', type=int, nargs=3,
              default=None, callback=default_none,
              help='save the big volume as chunks.')
@click.option('--compression', '-c', type=click.Choice(["gzip", "lzf", "szip"]),
              default="gzip", help="compression used in the dataset.")
@click.option('--with-offset/--without-offset', default=True, type=bool,
              help='add voxel_offset dataset or not.')
@click.option('--voxel-size', '-v',
    default=None, type=int, callback=default_none, nargs=3,
    help='voxel size of this chunk.'
)
@click.option('--touch/--no-touch', default=True, 
help = 'create an empty file if the input is None.'
)
@operator
def write_h5(tasks, input_name, file_name, chunk_size, compression, with_offset, voxel_size, touch):
    """Write chunk to HDF5 file."""
    for task in tasks:
        if task is not None:
            if not file_name.endswith('.h5'):
                bbox = task['bbox']
                file_name = f'{file_name}{bbox.to_filename()}.h5'
            data = task[input_name]
            if isinstance(data, Chunk):
                data.to_h5(
                    file_name, with_offset, 
                    chunk_size=chunk_size, 
                    compression=compression,
                    voxel_size=voxel_size)
            elif isinstance(data, Synapses):
                data.to_h5(file_name)
            elif data is None:
                if touch:
                    Path(file_name).touch()
            else:
                raise ValueError(f'unsuported type of input data: {data}')
        yield task


@main.command('write-pngs')
@click.option('--name', type=str, default='write-pngs', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--output-path', '-o',
              type=str, default='./pngs/', help='output path of saved 2d images formated as png.')
@operator
def write_pngs(tasks, name, input_chunk_name, output_path):
    """Save as 2D PNG images."""
    operator = WritePNGsOperator(output_path=output_path,
                                                name=name)
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
@click.option('--voxel-size', type=int, nargs=3, required=True,
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
                print('skip deleting task in queue!')
            else:
                queue = task['queue']
                task_handle = task['task_handle']
                queue.delete(task_handle)
                print('deleted task {} in queue: {}'.format(
                    task_handle, queue.queue_name))
        yield task


@main.command('delete-chunk')
@click.option('--name', type=str, default='delete-var', help='delete variable/chunk in task')
@click.option('--chunk-name', '-c',
              type=str, required=True, help='the chunk name need to be deleted')
@operator
def delete_chunk(tasks, name, chunk_name):
    """Delete a Chunk in task to release RAM"""
    for task in tasks:
        if task is not None:
            logging.info(f'delete chunk: {chunk_name}')
            del task[chunk_name]
        yield task
 

@main.command('read-precomputed')
@click.option('--name',
              type=str, default='read-precomputed', help='name of this operator')
@click.option('--volume-path', '-v',
              type=str, required=True, help='volume path')
@click.option('--mip', '-m',
              type=int, default=None, help='mip level of the cutout.')
@click.option('--expand-margin-size', '-e',
              type=int, nargs=3, default=(0, 0, 0),
              help='include surrounding regions of output bounding box.')
@click.option('--chunk-start', '-s',
              type=int, nargs=3, default=None, callback=default_none,
              help='chunk offset in volume.')
@click.option('--chunk-size', '-z',
              type=int, nargs=3, default=None, callback=default_none,
              help='cutout chunk size.')
@click.option('--fill-missing/--no-fill-missing',
              default=False, help='fill the missing chunks in input volume with zeros ' +
              'or not, default is false')
@click.option('--validate-mip', 
              type=int, default=None, help='validate chunk using higher mip level')
@click.option('--blackout-sections/--no-blackout-sections',
    default=False, help='blackout some sections. ' +
    'the section ids json file should named blackout_section_ids.json. default is False.')
@click.option(
    '--output-chunk-name', '-o',
    type=str, default=DEFAULT_CHUNK_NAME, 
    help='Variable name to store the cutout to for later retrieval.'
    + 'Chunkflow operators by default operates on a variable named "chunk" but' +
    ' sometimes you may need to have a secondary volume to work on.'
)
@operator
def read_precomputed(tasks, name, volume_path, mip, chunk_start, chunk_size, expand_margin_size,
           fill_missing, validate_mip, blackout_sections, output_chunk_name):
    """Cutout chunk from volume."""
    if mip is None:
        mip = state['mip']
    assert mip >= 0
    
    operator = ReadPrecomputedOperator(
        volume_path,
        mip=mip,
        expand_margin_size=expand_margin_size,
        fill_missing=fill_missing,
        validate_mip=validate_mip,
        blackout_sections=blackout_sections,
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
            assert output_chunk_name not in task
            task[output_chunk_name] = operator(bbox)
            task['log']['timer'][name] = time() - start
            task['cutout_volume_path'] = volume_path
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


@main.command('downsample-upload')
@click.option('--name',
              type=str, default='downsample-upload', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default='chunk', help='input chunk name')
@click.option('--volume-path', '-v', type=str, help='path of output volume')
@click.option('--factor', '-f', type=int, nargs=3, default=(2, 2, 2), 
    help='downsampling factor in z,y,x.')
@click.option('--chunk-mip', '-c', type=int, default=None, help='input chunk mip level')
@click.option('--start-mip', '-s', 
    type=int, default=None, help='the start uploading mip level.')
@click.option('--stop-mip', '-p',
    type=int, default=5, help='stop mip level. the indexing follows python style and ' +
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
    type=int, default=1, help='standard deviation of gaussian kernel')
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
    type=int, nargs=3, default=None, callback=default_none,
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


@main.command('normalize-contrast-nkem')
@click.option('--name', type=str, default='normalize-contrast-nkem',
              help='name of operator.')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--output-chunk-name', '-o',
              type=str, default=DEFAULT_CHUNK_NAME, help='output chunk name')
@click.option('--levels-path', '-p', type=str, required=True,
              help='the path of section histograms.')
@click.option('--lower-clip-fraction', '-l', type=float, default=0.01, 
              help='lower intensity fraction to clip out.')
@click.option('--upper-clip-fraction', '-u', type=float, default=0.01, 
              help='upper intensity fraction to clip out.')
@click.option('--minval', type=int, default=1, 
              help='the minimum intensity of transformed chunk.')
@click.option('--maxval', type=int, default=255,
              help='the maximum intensity of transformed chunk.')
@operator
def normalize_contrast_nkem(tasks, name, input_chunk_name, output_chunk_name, 
                                levels_path, lower_clip_fraction,
                                upper_clip_fraction, minval, maxval):
    """Normalize the section contrast using precomputed histograms."""
    
    operator = NormalizeSectionContrastOperator(
        levels_path,
        lower_clip_fraction=lower_clip_fraction,
        upper_clip_fraction=upper_clip_fraction,
        minval=minval, maxval=maxval, name=name)

    for task in tasks:
        if task is not None:
            start = time()
            task[output_chunk_name] = operator(task[input_chunk_name])
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
              type=float,
              default=None,
              help='targeted minimum of transformed chunk.')
@click.option('--nominalmax',
              type=float,
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

    operator = NormalizeSectionShangOperator(
        nominalmin=nominalmin,
        nominalmax=nominalmax,
        clipvalues=clipvalues,
        name=name)

    for task in tasks:
        if task is not None:
            start = time()
            task[output_chunk_name] = operator(task[input_chunk_name])
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
              help='arguments of plugin, this string should be interpreted inside plugin.')
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
                inputs = [task[i] for i in input_name_list]
            else:
                inputs = []
            outputs = operator(inputs, args=args)
            if outputs is not None:
                output_name_list = output_names.split(',')
                assert len(outputs) == len(output_name_list)
                for output_name, output in zip(output_name_list, outputs):
                    task[output_name] = output
            else:
                assert output_names is None

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
@click.option('--threshold', '-t', type=float, default=None,
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
              type=int, nargs=3, required=True, help='input patch size')
@click.option('--output-patch-size', '-z', type=int, nargs=3, default=None, 
              callback=default_none, help='output patch size')
@click.option('--output-patch-overlap', '-v', type=int, nargs=3, 
              default=(4, 64, 64), help='patch overlap')
@click.option('--output-crop-margin', type=int, nargs=3,
              default=None, callback=default_none, help='margin size of output cropping.')
@click.option('--patch-num', '-n', default=None, callback=default_none,
              type=int, nargs=3, help='patch number in z,y,x.')
@click.option('--num-output-channels', '-c',
              type=int, default=3, help='number of output channels')
@click.option('--dtype', '-d', type=click.Choice(['float32', 'float16']),
              default='float32', help="""Even if we perform inference using float16, 
                    the result will still be converted to float32.""")
@click.option('--framework', '-f',
              type=click.Choice(['universal', 'identity', 'pytorch']),
              default='universal', help='inference framework')
@click.option('--batch-size', '-b',
              type=int, default=1, help='mini batch size of input patch.')
@click.option('--bump', type=click.Choice(['wu', 'zung']), default='wu',
              help='bump function type (only support wu now!).')
@click.option('--mask-output-chunk/--no-mask-output-chunk', default=False,
              help='mask output chunk will make the whole chunk like one output patch. '
              + 'This will also work with non-aligned chunk size.')
@click.option('--mask-myelin-threshold', '-y', default=None, type=float,
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


@main.command('mask')
@click.option('--name', type=str, default='mask', help='name of this operator')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--output-chunk-name', '-o',
              type=str, default=DEFAULT_CHUNK_NAME, help='output chunk name')
@click.option('--volume-path', '-v',
              type=str, required=True, help='mask volume path')
@click.option('--mip', '-m', 
              type=int, default=5, help='mip level of mask')
@click.option('--inverse/--no-inverse',
              default=False,
              help='inverse the mask or not. default is True. ' +
              'the mask will be multiplied to chunk.')
@click.option('--fill-missing/--no-fill-missing',
              default=False,
              help='fill missing blocks with black or not. ' +
              'default is False.')
@operator
def mask(tasks, name, input_chunk_name, output_chunk_name, volume_path, 
         mip, inverse, fill_missing):
    """Mask the chunk. The mask could be in higher mip level and we
    will automatically upsample it to the same mip level with chunk.
    """
    operator = MaskOperator(volume_path,
                            mip,
                            state['mip'],
                            inverse=inverse,
                            fill_missing=fill_missing,
                            name=name)

    for task in tasks:
        if task is not None:
            start = time()
            task[output_chunk_name] = operator(task[input_chunk_name])
            # Note that mask operation could be used several times,
            # this will only record the last masking operation
            task['log']['timer'][name] = time() - start
        yield task


@main.command('mask-out-objects')
@click.option('--input-chunk-name', '-i', type=str, default=DEFAULT_CHUNK_NAME)
@click.option('--output_chunk_name', '-o', type=str, default=DEFAULT_CHUNK_NAME)
@click.option('--dust-size-threshold', '-d', type=int, default=None,
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
        json_storage = Storage(os.path.dirname(selected_obj_ids))
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
@click.option('--name',
              type=str,
              default='crop-margin',
              help='name of this operator')
@click.option('--margin-size', '-m',
              type=int, nargs=3, default=None, callback=default_none,
              help='crop the chunk margin. ' +
              'The default is None and will use the bbox as croping range.')
@click.option('--input-chunk-name', '-i',
              type=str, default='chunk', help='input chunk name.')
@click.option('--output-chunk-name', '-o',
              type=str, default='chunk', help='output chunk name.')
@operator
def crop_margin(tasks, name, margin_size, 
                input_chunk_name, output_chunk_name):
    """Crop the margin of chunk."""
    for task in tasks:
        if task is not None:
            start = time()
            if margin_size:
                task[output_chunk_name] = task[input_chunk_name].crop_margin(
                    margin_size=margin_size)
            else:
                # use the output bbox for croping 
                task[output_chunk_name] = task[
                    input_chunk_name].cutout(task['bbox'].to_slices())
            task['log']['timer'][name] = time() - start
        yield task


@main.command('mesh')
@click.option('--name', type=str, default='mesh', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='name of chunk needs to be meshed.')
@click.option('--mip', '-m',
    type=int, default=None, help='mip level of segmentation chunk.')
@click.option('--voxel-size', '-v', type=int, nargs=3, default=None, callback=default_none, 
    help='voxel size of the segmentation. zyx order.')
@click.option('--output-path', '-o', type=str, default='file:///tmp/mesh/', 
    help='output path of meshes, follow the protocol rule of CloudVolume. \
              The path will be adjusted if there is a info file with precomputed format.')
@click.option('--output-format', '-t', type=click.Choice(['ply', 'obj', 'precomputed']), 
              default='precomputed', help='output format, could be one of ply|obj|precomputed.')
@click.option('--simplification-factor', '-f', type=int, default=100, 
              help='mesh simplification factor.')
@click.option('--max-simplification-error', '-e', type=int, default=40, 
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
@click.option('--name', type=str, default='mesh-manifest', help='name of operator')
@click.option('--input-name', '-i', type=str, default='prefix', help='input key name in task.')
@click.option('--prefix', '-p', type=int, default=None, help='prefix of meshes.')
@click.option('--disbatch/--no-disbatch', default=False, help='use disBatch task index as prefix')
@click.option('--digits', '-d', type=int, default=1, help='number of digits of prefix')
@click.option('--volume-path', '-v', type=str, required=True, help='cloudvolume path of dataset layer.' + 
              ' The mesh directory will be automatically figure out using the info file.')
@operator
def mesh_manifest(tasks, name, input_name, prefix, disbatch, digits, volume_path):
    """Generate mesh manifest files."""
    operator = MeshManifestOperator(volume_path)
    if prefix:
        operator(prefix, digits)
    elif disbatch:
            assert 'DISBATCH_REPEAT_INDEX' in os.environ
            prefix = os.environ['DISBATCH_REPEAT_INDEX']
            operator(prefix, digits)
    elif input_name:
        for task in tasks:
            start = time()
            operator(task[input_name], digits)
            task['log']['timer'][name] = time() - start
            yield task
    else:
        logging.error('requires one of parameters: prefix, input_name, disbatch')


@main.command('neuroglancer')
@click.option('--name', type=str, default='neuroglancer',
              help='name of this operator')
@click.option('--voxel-size', '-v',
              nargs=3, type=int, default=None, callback=default_none,
              help='voxel size of chunk')
@click.option('--port', '-p', type=int, default=None, help='port to use')
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
@click.option('--name', type=str, default='quantize', help='name of this operator')
@click.option('--input-chunk-name', '-i', type=str, default='chunk', help = 'input chunk name')
@click.option('--output-chunk-name', '-o', type=str, default='chunk', help= 'output chunk name')
@operator
def quantize(tasks, name, input_chunk_name, output_chunk_name):
    """Transorm the last channel to uint8."""
    for task in tasks:
        if task is not None:
            aff = task[input_chunk_name]
            aff = AffinityMap(aff)
            assert isinstance(aff, AffinityMap)
            quantized_image = aff.quantize()
            task[output_chunk_name] = quantized_image
        yield task

@main.command('write-precomputed')
@click.option('--name', type=str, default='write-precomputed', help='name of this operator')
@click.option('--volume-path', '-v', type=str, required=True, help='volume path')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--mip', '-m',
    type=int, default=None, help="mip level to write")
@click.option('--upload-log/--no-upload-log',
              default=False, help='the log will be put inside volume-path')
@click.option('--create-thumbnail/--no-create-thumbnail',
    default=False, help='create thumbnail or not. ' +
    'the thumbnail is a downsampled and quantized version of the chunk.')
@click.option('--intensity-threshold', '-t',
    default=None, type=float,
    help='do not save anything if all voxel intensity is below threshold.'
)
@operator
def write_precomputed(tasks, name: str, volume_path: str, 
        input_chunk_name: str, mip: int, upload_log: bool, 
        create_thumbnail: bool, intensity_threshold: float):
    """Save chunk to volume."""
    if mip is None:
        mip = state['mip']

    operator = WritePrecomputedOperator(
        volume_path,
        mip,
        upload_log=upload_log,
        create_thumbnail=create_thumbnail,
        name=name
    )

    for task in tasks:
        if task is not None:
            # the time elapsed was recorded internally
            chunk = task[input_chunk_name]
            if intensity_threshold is not None and np.all(chunk.array < intensity_threshold):
                pass
            else:
                operator(chunk, log=task.get('log', {'timer': {}}))
                task['output_volume_path'] = volume_path

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
@click.option('--threshold', '-t', type=float, default=0.5,
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
@click.option('--name', type=str, default='channel-voting', help='name of operator')
@click.option('--input-chunk-name', type=str, default=DEFAULT_CHUNK_NAME)
@click.option('--output-chunk-name', type=str, default=DEFAULT_CHUNK_NAME)
@operator
def channel_voting(tasks, name, input_chunk_name, output_chunk_name):
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
