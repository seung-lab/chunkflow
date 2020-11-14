#!/usr/bin/env python
import os
from time import time

import numpy as np
import click

from .lib import *

from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec, yellow

from chunkflow.lib.aws.sqs_queue import SQSQueue
from chunkflow.lib.bounding_boxes import BoundingBoxes 

from chunkflow.chunk import Chunk
from chunkflow.chunk.affinity_map import AffinityMap
from chunkflow.chunk.segmentation import Segmentation
from chunkflow.chunk.image.convnet.inferencer import Inferencer

# import operator functions
from .agglomerate import AgglomerateOperator
from .aggregate_skeleton_fragments import AggregateSkeletonFragmentsOperator
from .cloud_watch import CloudWatchOperator
from .read_precomputed import ReadPrecomputedOperator
from .downsample_upload import DownsampleUploadOperator
from .log_summary import load_log, print_log_statistics
from .mask import MaskOperator
from .mask_out_objects import MaskOutObjectsOperator
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
              type=int, default=0, help='mip level of the dataset layer.')
@click.option('--roi-start', '-s',
              type=int, default=None, nargs=3, callback=default_none, 
              help='(z y x), start of the chunks')
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
@generator
def generate_tasks(layer_path, mip, roi_start, chunk_size, 
                   grid_size, file_path, queue_name):
    """Generate tasks."""
    bboxes = BoundingBoxes.from_manual_setup(
        chunk_size, layer_path=layer_path,
        roi_start=roi_start, mip=mip, grid_size=grid_size,
        verbose=state['verbose'])

    print('total number of tasks: ', len(bboxes)) 
    # write out as a file
    # this could be used for iteration in slurm cluster.
    if file_path:
        if not file_path.endswith('.npy'):
            file_path += len(bboxes) + '.npy'
        bboxes.to_file(file_path)

    if queue_name is not None:
        queue = SQSQueue(queue_name)
        queue.send_message_list(bboxes)
    else:
        for bbox in bboxes:
            task = get_initial_task()
            task['bbox'] = bbox
            task['log']['bbox'] = bbox.to_filename()
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

    bboxes = setup_environment(
        state['dry_run'], volume_start, volume_stop, volume_size, layer_path, 
        max_ram_size, output_patch_size, input_patch_size, channel_num, dtype, 
        output_patch_overlap, crop_chunk_margin, mip, thumbnail_mip, max_mip,
        thumbnail, encoding, voxel_size, overwrite_info, state['verbose'])
 
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
    operator = CloudWatchOperator(log_name=log_name,
                                                  name=name,
                                                  verbose=state['verbose'])
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            operator(task['log'])
        yield task


@main.command('create-info')
@click.option('--layer-path', '-l', type=str, required=True, help='path of layer.')
@click.option('--channel-num', '-c', type=int, default=1, help='number of channel')
@click.option('--layer-type', '-t',
              type=click.Choice(['image', 'segmentation']),
              default='image', help='type of layer. either image or segmentation.')
@click.option('--data-type', '-d',
              type=click.Choice(['uint8', 'uint32', 'uint64', 'float32']),
              required=True, help='data type of array')
@click.option('--encoding', '-e',
              type=click.Choice(['raw', 'jpeg', 'compressed_segmentation', 
                    'kempressed', 'npz', 'fpzip', 'npz_uint8']),
              default='raw', help='compression algorithm.')
@click.option('--voxel-size', '-s', required=True, type=int, nargs=3,
              help='voxel size with unit of nm')
@click.option('--voxel-offset', '-o', default=(0,0,0), type=int, nargs=3,
              help='voxel offset of array')
@click.option('--volume-size', '-v',
              required=True, type=int, nargs=3,
              help='total size of the volume.')
@click.option('--block-size', '-b',
              type=int, nargs=3, required=True,
              help='chunk size of each file.')
@click.option('--max-mip', '-m',
              type=int, default=0, 
              help = 'maximum mip level.')
@operator
def create_info(tasks,layer_path, channel_num, layer_type, data_type, encoding, voxel_size, 
                voxel_offset, volume_size, block_size, max_mip):
    info = CloudVolume.create_new_info(
        channel_num, layer_type=layer_type,
        data_type=data_type,
        encoding=encoding,
        resolution=voxel_size[::-1],
        voxel_offset=voxel_offset[::-1],
        volume_size=volume_size[::-1],
        chunk_size=block_size[::-1],
        max_mip=max_mip)
    vol = CloudVolume(layer_path, info=info)
    vol.commit_info()
    return info


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
    if(slurm_job_array):
        job_index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    assert job_index is not None

    bbox_array = np.load(file_path)
    task_start = job_index * granularity 
    task_stop = min(bbox_array.shape[0], task_start + granularity)
    for idx in range(task_start, task_stop):
        bbox = Bbox.from_list(bbox_array[idx, :])
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
        bbox = Bbox.from_filename(bbox_str)
        
        # record the task handle to delete after the processing
        task = get_initial_task() 
        task['queue'] = queue
        task['task_handle'] = task_handle
        task['bbox'] = bbox
        task['log']['bbox'] = bbox.to_filename()
        yield task


@main.command('agglomerate')
@click.option('--name', type=str, default='agglomerate', help='name of operator')
@click.option('--threshold', '-t',
              type=float, default=0.7, help='agglomeration threshold')
@click.option('--aff-threshold-low', '-l',
              type=float, default=0.0001, help='low threshold for watershed')
@click.option('--aff-threshold-high', '-h',
              type=float, default=0.9999, help='high threshold for watershed')
@click.option('--fragments-chunk-name', '-f',
              type=str, default=None, help='optional fragments/supervoxel chunk to use.')
@click.option('--scoring-function', '-s',
              type=str, default='OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
              help='A C++ type string specifying the edge scoring function to use.')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--output-chunk-name', '-o',
              type=str, default=DEFAULT_CHUNK_NAME, help='output chunk name')
@operator
def agglomerate(tasks, name, threshold, aff_threshold_low, aff_threshold_high,
                fragments_chunk_name, scoring_function, input_chunk_name, output_chunk_name):
    """Watershed and agglomeration to segment affinity map."""
    operator = AgglomerateOperator(name=name, verbose=state['verbose'],
                                                   threshold=threshold, 
                                                   aff_threshold_low=aff_threshold_low,
                                                   aff_threshold_high=aff_threshold_high,
                                                   scoring_function=scoring_function)
    for task in tasks:
        if fragments_chunk_name and fragments_chunk_name in task:
            fragments = task[fragments_chunk_name]
        else:
            fragments = None 
        
        task[output_chunk_name] = operator(
            task[input_chunk_name], fragments=fragments)
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
@click.option('--voxel-offset',
              type=int, nargs=3, default=(0, 0, 0), help='offset in voxel number.')
@click.option('--output-chunk-name', '-o',
              type=str, default="chunk", help="name of created chunk")
@operator
def create_chunk(tasks, name, size, dtype, voxel_offset, all_zero, output_chunk_name):
    """Create a fake chunk for easy test."""
    print("creating chunk: ", output_chunk_name)
    for task in tasks:
        task[output_chunk_name] = Chunk.create(
            size=size, dtype=np.dtype(dtype), 
            all_zero = all_zero,
            voxel_offset=voxel_offset)
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
@click.option('--chunk-size', '-s',
              type=int, nargs=3, default=None, callback=default_none,
              help='cutout chunk size')
@operator
def read_pngs(tasks, path_prefix, output_chunk_name, cutout_offset, volume_offset, chunk_size):
    """Read a serials of png files."""
    for task in tasks:
        if chunk_size is None:
            assert 'bbox' in task, "no chunk_size, we are looking for bounding box in task"
            bbox = task['bbox']
        else:
            bbox = Bbox.from_delta(cutout_offset, chunk_size)

        task[output_chunk_name] = read_png_images(
            path_prefix, bbox, 
            volume_offset=volume_offset,
            verbose = state['verbose'])
        yield task


@main.command('read-tif')
@click.option('--name', type=str, default='read-tif',
              help='read tif file from local disk.')
@click.option('--file-name', '-f', required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='read chunk from file, support .h5 and .tif')
@click.option('--voxel-offset', '-v', type=int, nargs=3, callback=default_none,
              help='global offset of this chunk')
@click.option('--dtype', '-d',
              type=click.Choice(['uint8', 'uint32', 'uint64', 'float32', 'float64', 'float16']),
              help='convert to data type')
@click.option('--output-chunk-name', '-o', type=str, default='chunk',
              help='chunk name in the global state')
@operator
def read_tif(tasks, name: str, file_name: str, voxel_offset: tuple,
             dtype: str, output_chunk_name: str):
    """Read tiff files."""
    for task in tasks:
        start = time()
        assert output_chunk_name not in task
        task[output_chunk_name] = Chunk.from_tif(file_name,
                                                    dtype=dtype,
                                                    voxel_offset=voxel_offset)
        task['log']['timer'][name] = time() - start
        yield task


@main.command('read-h5')
@click.option('--name', type=str, default='read-h5',
              help='read file from local disk.')
@click.option('--file-name', '-f', type=str, required=True,
              help='read chunk from file, support .h5')
@click.option('--dataset-path', '-d', type=str, default=None, callback=default_none,
              help='the dataset path inside HDF5 file.')
@click.option('--voxel-offset', '-v', type=int, nargs=3,
              callback=default_none, help='voxel offset of the dataset in hdf5 file.')
@click.option('--cutout-start', '-t', type=int, nargs=3, callback=default_none,
              help='cutout voxel offset in the array')
@click.option('--cutout-stop', '-p', type=int, nargs=3, callback=default_none,
               help='cutout stop corrdinate.')
@click.option('--cutout-size', '-s', type=int, nargs=3, callback=default_none,
               help='cutout size of the chunk.')
@click.option('--output-chunk-name', '-o',
              type=str, default='chunk',
              help='chunk name in the global state')
@operator
def read_h5(tasks, name: str, file_name: str, dataset_path: str,
            voxel_offset: tuple, cutout_start: tuple, 
            cutout_stop: tuple, cutout_size: tuple, output_chunk_name: str):
    """Read HDF5 files."""
    for task in tasks:
        
        start = time()
        if 'bbox' in task and cutout_start is None and cutout_stop is None and cutout_size is None:
            bbox = task['bbox']
            print('bbox: ', bbox) 
            current_cutout_start = bbox.minpt
            current_cutout_stop = bbox.maxpt
        else:
            current_cutout_start = cutout_start
            current_cutout_stop = cutout_stop
        
        print(f'cutout start: {current_cutout_start}')
        print(f'cutout stop: {current_cutout_stop}')
        
        task[output_chunk_name] = Chunk.from_h5(
            file_name,
            dataset_path=dataset_path,
            voxel_offset=voxel_offset,
            cutout_start=current_cutout_start,
            cutout_stop=current_cutout_stop,
            cutout_size=cutout_size
        )
        task['log']['timer'][name] = time() - start
        yield task


@main.command('write-h5')
@click.option('--name', type=str, default='write-h5', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default='chunk', help='input chunk name')
@click.option('--file-name',
              '-f',
              type=click.Path(dir_okay=True, resolve_path=True),
              required=True,
              help='file name of hdf5 file.')
@click.option('--compression', '-c', type=click.Choice(["gzip", "lzf", "szip"]),
              default="gzip", help="compression used in the dataset.")
@click.option('--with-offset/--without-offset', default=True, type=bool,
              help='add voxel_offset dataset or not.')
@operator
def write_h5(tasks, name, input_chunk_name, file_name, compression, with_offset):
    """Write chunk to HDF5 file."""
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            task[input_chunk_name].to_h5(file_name, with_offset, compression=compression)
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
        handle_task_skip(task, name)
        if not task['skip']:
            task[input_chunk_name].to_tif(file_name)
        # keep the pipeline going
        yield task


@main.command('write-pngs')
@click.option('--name', type=str, default='write-pngs', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, help='input chunk name')
@click.option('--output-path', '-o',
              type=str, default='./saved_pngs/', help='output path of saved 2d images formated as png.')
@operator
def write_pngs(tasks, name, input_chunk_name, output_path):
    """Save as 2D PNG images."""
    operator = WritePNGsOperator(output_path=output_path,
                                                name=name)
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
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
    operator = SkeletonizeOperator(output_path,
                                   name=name,
                                   verbose=state['verbose'])
    for task in tasks:
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
        handle_task_skip(task, name)
        if task['skip'] or state['dry_run']:
            print('skip deleting task in queue!')
        else:
            queue = task['queue']
            task_handle = task['task_handle']
            queue.delete(task_handle)
            print('deleted task {} in queue: {}'.format(
                task_handle, queue.queue_name))


@main.command('delete-chunk')
@click.option('--name', type=str, default='delete-var', help='delete variable/chunk in task')
@click.option('--chunk-name', '-c',
              type=str, required=True, help='the chunk name need to be deleted')
@operator
def delete_chunk(tasks, name, chunk_name):
    """Delete a Chunk in task to release RAM"""
    for task in tasks:
        handle_task_skip(task, name)
        if task['skip']:
            print('skip deleting ', chunk_name)
        else:
            if state['verbose']:
                print('delete chunk: ', chunk_name)
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
@click.option('--output-chunk-name', '-o',
    type=str, default='chunk', help='Variable name to store the cutout to for later retrieval.'
    + 'Chunkflow operators by default operates on a variable named "chunk" but' +
    ' sometimes you may need to have a secondary volume to work on.')
@operator
def read_precomputed(tasks, name, volume_path, mip, chunk_start, chunk_size, expand_margin_size,
           fill_missing, validate_mip, blackout_sections, output_chunk_name):
    """Cutout chunk from volume."""
    if mip is None:
        mip = state['mip']
    
    operator = ReadPrecomputedOperator(
        volume_path,
        mip=mip,
        expand_margin_size=expand_margin_size,
        verbose=state['verbose'],
        fill_missing=fill_missing,
        validate_mip=validate_mip,
        blackout_sections=blackout_sections,
        dry_run=state['dry_run'],
        name=name)

    for task in tasks:
        handle_task_skip(task, name)
        if 'bbox' in task:
            bbox = task['bbox']
        else:
            # use bounding box of volume
            if chunk_start is None:
                chunk_start = operator.vol.mip_bounds(mip).minpt
            else:
                chunk_start = Vec(*chunk_start)

            if chunk_size is None:
                chunk_stop = operator.vol.mip_bounds(mip).maxpt
                chunk_size = chunk_stop - chunk_start
            else:
                chunk_size = Vec(*chunk_size)
            bbox = Bbox.from_delta(chunk_start, chunk_size)

        if not task['skip']:
            start = time()
            assert output_chunk_name not in task
            task[output_chunk_name] = operator(bbox)
            task['log']['timer'][name] = time() - start
            task['cutout_volume_path'] = volume_path
        yield task


@main.command('evaluate-segmentation')
@click.option('--name',
              type=str,
              default="evaluate-segmentation",
              help="name of operator")
@click.option("--segmentation-chunk-name",
              "-s",
              type=str,
              default="chunk",
              help="chunk name of segmentation")
@click.option("--groundtruth-chunk-name",
              "-g",
              type=str,
              default="groundtruth")
@operator
def evaluate_segmenation(tasks, name, segmentation_chunk_name,
                         groundtruth_chunk_name):
    """Evaluate segmentation by split/merge error.
    """
    for task in tasks:
        seg = Segmentation(task[segmentation_chunk_name])
        groundtruth = Segmentation(task[groundtruth_chunk_name])
        seg.evaluate(groundtruth)
        yield task


@main.command('downsample-upload')
@click.option('--name',
              type=str, default='downsample-upload', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default='chunk', help='input chunk name')
@click.option('--volume-path', '-v', type=str, help='path of output volume')
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
                      chunk_mip, start_mip, stop_mip, fill_missing):
    """Downsample chunk and upload to volume."""
    if chunk_mip is None:
        chunk_mip = state['mip']

    operator = DownsampleUploadOperator(
        volume_path,
        chunk_mip=chunk_mip,
        start_mip=start_mip,
        stop_mip=stop_mip,
        fill_missing=fill_missing,
        name=name,
        verbose=state['verbose'])

    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            operator(task[input_chunk_name])
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
        handle_task_skip(task, name)
        if not task['skip']:
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
        handle_task_skip(task, name)
        if not task['skip']:
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
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            task[output_chunk_name] = operator(task[input_chunk_name])
            task['log']['timer'][name] = time() - start
        yield task


@main.command('plugin')
@click.option('--name',
              type=str,
              default='plugin-1',
              help='name of plugin. Multiple plugins should have different names.')
@click.option('--input-chunk-name', '-i',
              type=str, default='chunk', help='input chunk name')
@click.option('--output-chunk-name', '-o',
              type=str, default='chunk', help='output chunk name')
@click.option('--file', '-f', type=str, help='''python file to call. 
                If it is just a name rather than full path, 
                we\'ll look for it in the plugin folder.''')
@click.option('--args', type=str, default='', help='args to pass in')
@operator
def plugin(tasks, name, input_chunk_name, output_chunk_name, file, args):
    """Insert custom program as a plugin.
    The custom python file should contain a callable named "exec" such that 
    a call of `exec(chunk, args)` can be made to operate on the chunk.
    """

    operator = Plugin(file, args=args, name=name, verbose=state['verbose'])
    if state['verbose']:
        print('Received args for ', name, ':', args)

    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            task[output_chunk_name] = operator(task[input_chunk_name], *args)
            task['log']['timer'][name] = time() - start
        yield task


@main.command('connected-components')
@click.option('--name', type=str, default='connected-components', 
              help='threshold a map and get the labels.')
@click.option('--input-chunk-name', '-i',
              type=str, default=DEFAULT_CHUNK_NAME, 
              help='input chunk name')
@click.option('--output-chunk-name', '-o',
              type=str, default=DEFAULT_CHUNK_NAME, 
              help='output chunk name')
@click.option('--threshold', '-t', type=float, default=0.5,
              help='threshold to cut the map.')
@click.option('--connectivity', '-c', 
              type=click.Choice(['6', '18', '26']),
              default='26', help='number of neighboring voxels used.')
@operator 
def connected_components(tasks, name, input_chunk_name, output_chunk_name, 
                         threshold, connectivity):
    """Threshold the probability map to get a segmentation."""
    connectivity = int(connectivity)
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            task[output_chunk_name] = task[input_chunk_name].connected_component(
                threshold=threshold, connectivity=connectivity)
            task['log']['timer']['name'] = time() - start
        yield task


@main.command('copy-var')
@click.option('--name', type=str, default='copy-var-1', help='name of step')
@click.option('--from-name',
              type=str,
              default='chunk',
              help='Variable to be (shallow) copied/"renamed"')
@click.option('--to-name', type=str, default='chunk', help='New variable name')
@operator
def copy_var(tasks, name, from_name, to_name):
    """Copy a variable to a new name.
    """
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
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
        dry_run=state['dry_run'],
        verbose=state['verbose']) as inferencer:
        
        for task in tasks:
            handle_task_skip(task, name)
            if not task['skip']:
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
@click.option('--check-all-zero/--maskout',
              default=False,
              help='default is doing maskout. ' +
              'check all zero will return boolean result.')
@click.option('--skip-to', type=str, default='write-precomputed', help='skip to a operator')
@operator
def mask(tasks, name, input_chunk_name, output_chunk_name, volume_path, 
         mip, inverse, fill_missing, check_all_zero, skip_to):
    """Mask the chunk. The mask could be in higher mip level and we
    will automatically upsample it to the same mip level with chunk.
    """
    operator = MaskOperator(volume_path,
                                            mip,
                                            state['mip'],
                                            inverse=inverse,
                                            fill_missing=fill_missing,
                                            check_all_zero=check_all_zero,
                                            verbose=state['verbose'],
                                            name=name)

    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            if check_all_zero:
                # skip following operators since the mask is all zero after required inverse
                task['skip'] = operator.is_all_zero(
                    task['bbox'])
                if task['skip']:
                    print(yellow(f'the mask of {name} is all zero, will skip to {skip_to}'))
                task['skip_to'] = skip_to
            else:
                task[output_chunk_name] = operator(task[input_chunk_name])
            # Note that mask operation could be used several times,
            # this will only record the last masking operation
            task['log']['timer'][name] = time() - start
        yield task


@main.command('mask-out-objects')
@click.option('--name', '-n', type=str, default='mask-out-objects',
              help='remove some objects in segmentation chunk.')
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
def mask_out_objects(tasks, name, input_chunk_name, output_chunk_name,
                     dust_size_threshold, selected_obj_ids):
    """Mask out objects in a segmentation chunk."""
    operator = MaskOutObjectsOperator(
        dust_size_threshold,
        selected_obj_ids,
        name=name,
        verbose=state['verbose']
    )
    operator = operator
    
    for task in tasks:
        task[output_chunk_name] = operator(task[input_chunk_name])
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
        handle_task_skip(task, name)
        if not task['skip']:
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
@click.option('--manifest/--no-manifest', default=False, help='create manifest file or not.')
@operator
def mesh(tasks, name, input_chunk_name, mip, voxel_size, output_path, output_format,
         simplification_factor, max_simplification_error, manifest):
    """Perform meshing for segmentation chunk."""
    if mip is None:
        mip = state['mip']

    operator = MeshOperator(
        output_path,
        output_format,
        mip=mip,
        voxel_size=voxel_size,
        simplification_factor=simplification_factor,
        max_simplification_error=max_simplification_error,
        manifest=manifest)
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            operator( task[input_chunk_name] )
            task['log']['timer'][name] = time() - start
        yield task

@main.command('mesh-manifest')
@click.option('--name', type=str, default='mesh-manifest', help='name of operator')
@click.option('--input-name', '-i', type=str, default='prefix', help='input key name in task.')
@click.option('--prefix', '-p', type=str, default=None, help='prefix of meshes.')
@click.option('--volume-path', '-v', type=str, required=True, help='cloudvolume path of dataset layer.' + 
              ' The mesh directory will be automatically figure out using the info file.')
@operator
def mesh_manifest(tasks, name, input_name, prefix, volume_path):
    """Generate mesh manifest files."""
    operator = MeshManifestOperator(volume_path)
    if prefix:
        operator(prefix)
    else:
        for task in tasks:
            handle_task_skip(task, name)
            if not task['skip']:
                start = time()
                operator(task[input_name])
                task['log']['timer'][name] = time() - start
            yield task
 
@main.command('neuroglancer')
@click.option('--name',
              type=str,
              default='neuroglancer',
              help='name of this operator')
@click.option('--voxel-size',
              '-v',
              nargs=3,
              type=int,
              default=(1, 1, 1),
              help='voxel size of chunk')
@click.option('--port', '-p', type=int, default=None, help='port to use')
@click.option('--chunk-names', '-c', type=str, default='chunk', 
              help='a list of chunk names separated by comma.')
@operator
def neuroglancer(tasks, name, voxel_size, port, chunk_names):
    """Visualize the chunk using neuroglancer."""
    operator = NeuroglancerOperator(name=name,
                                                    port=port,
                                                    voxel_size=voxel_size)
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            operator(task, selected=chunk_names)
        yield task


@main.command('quantize')
@click.option('--name', type=str, default='quantize', help='name of this operator')
@click.option('--input-chunk-name', '-i', type=str, default='chunk', help = 'input chunk name')
@click.option('--output-chunk-name', '-o', type=str, default='chunk', help= 'output chunk name')
@operator
def quantize(tasks, name, input_chunk_name, output_chunk_name):
    """Transorm the last channel to uint8."""
    for task in tasks:
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
@click.option('--upload-log/--no-upload-log',
              default=True, help='the log will be put inside volume-path')
@click.option('--create-thumbnail/--no-create-thumbnail',
    default=False, help='create thumbnail or not. ' +
    'the thumbnail is a downsampled and quantized version of the chunk.')
@operator
def write_precomputed(tasks, name, volume_path, input_chunk_name, upload_log, create_thumbnail):
    """Save chunk to volume."""
    operator = WritePrecomputedOperator(volume_path,
                                            state['mip'],
                                            upload_log=upload_log,
                                            create_thumbnail=create_thumbnail,
                                            verbose=state['verbose'],
                                            name=name)

    for task in tasks:
        # we got a special case for handling skip
        if task['skip'] and task['skip_to'] == name:
            task['skip'] = False
            # create fake chunk to save
            task[input_chunk_name] = operator.create_chunk_with_zeros(
                task['bbox'])

        if not task['skip']:
            # the time elapsed was recorded internally
            operator(task[input_chunk_name],
                                     log=task.get('log', {'timer': {}}))
            task['output_volume_path'] = volume_path
        yield task


@main.command('threshold')
@click.option('--name', type=str, default='threshold', 
              help='threshold a map and get the labels.')
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
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            if state['verbose']:
                print('Segment probability map using a threshold...')
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
        handle_task_skip(task, name)
        if not task['skip']:
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
        handle_task_skip(task, name)
        if not task['skip']:
            operator(task[image_chunk_name],
                                     seg=segmentation_chunk_name)
        yield task



if __name__ == '__main__':
    main()
