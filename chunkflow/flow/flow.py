#!/usr/bin/env python
import click
from functools import wraps, update_wrapper
from time import time
import numpy as np
from cloudvolume.lib import Bbox

from chunkflow.lib.aws.sqs_queue import SQSQueue
from chunkflow.chunk.segmentation import Segmentation

# import operator functions
from .cloud_watch import CloudWatchOperator
from .create_chunk import CreateChunkOperator
from .crop_margin import CropMarginOperator
from .custom_operator import CustomOperator
from .connected_components import ConnectedComponentsOperator
from .cutout import CutoutOperator
from .downsample_upload import DownsampleUploadOperator
from .inference import InferenceOperator
from .mask import MaskOperator
from .mesh import MeshOperator
from .neuroglancer import NeuroglancerOperator
from .normalize_section_contrast import NormalizeSectionContrastOperator
from .normalize_section_shang import NormalizeSectionShangOperator
from .read_tif import ReadTIFOperator
from .read_h5 import ReadH5Operator
from .save import SaveOperator
from .save_pngs import SavePNGsOperator
from .view import ViewOperator
from .write_h5 import WriteH5Operator
from .write_tif import WriteTIFOperator

# global dict to hold the operators and parameters
state = {'operators': {}}
INITIAL_TASK = {'skip': False, 'log': {'timer': {}}}


def handle_task_skip(task, name):
    if task['skip'] and task['skip_to'] == name:
        # have already skipped to target operator
        task['skip'] = False


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
@click.option('--verbose/--quiet',
              default=True,
              help='print informations or not, default is verbose.')
@click.option('--mip',
              type=int,
              default=0,
              help='default mip level of chunks.')
# the code design is from:
# https://github.com/pallets/click/blob/master/examples/imagepipe/imagepipe.py
def main(verbose, mip):
    """Compose operators and create your own pipeline."""
    state['verbose'] = verbose
    state['mip'] = mip
    pass


@main.resultcallback()
def process_commands(operators, verbose, mip):
    """This result callback is invoked with an iterable of all 
    the chained subcommands. As in this example each subcommand 
    returns a function we can chain them together to feed one 
    into the other, similar to how a pipe on unix works.
    """
    # It turns out that a tuple will not work correctly!
    stream = [
        INITIAL_TASK,
    ]

    # Pipe it through all stream operators.
    for operator in operators:
        stream = operator(stream)

    # Evaluate the stream and throw away the items.
    if stream:
        for _ in stream:
            pass


def operator(func):
    """Help decorator to rewrite a function so that it returns another function from it."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        def operator(stream):
            return func(stream, *args, **kwargs)

        return operator

    return wrapper


def generator(func):
    """Similar to the :func:`operator` but passes through old values unchanged 
    and does not pass through the values as parameter.
    """
    @operator
    def new_func(stream, *args, **kwargs):
        for item in stream:
            yield item
        for item in func(*args, **kwargs):
            yield item

    return update_wrapper(new_func, func)


@main.command('generate-task')
@click.option('--offset',
              type=int,
              nargs=3,
              default=(0, 0, 0),
              help='output offset')
@click.option('--shape',
              type=int,
              nargs=3,
              default=(0, 0, 0),
              help='output shape')
@generator
def generate_task(offset, shape):
    """Create a task."""
    # no queue name specified
    # will only run one task
    stop = np.asarray(offset) + np.asarray(shape)
    output_bbox = Bbox.from_list([*offset, *stop])
    print("creating task: ", output_bbox.to_filename())
    task = INITIAL_TASK
    task['output_bbox'] = output_bbox
    task['log']['output_bbox'] = output_bbox.to_filename()
    yield task


@main.command('fetch-task')
@click.option('--queue-name', type=str, default=None, help='sqs queue name')
@click.option(
    '--visibility-timeout',
    type=int,
    default=None,
    help=
    'visibility timeout of sqs queue; default is using the timeout of the queue.'
)
@generator
def fetch_task(queue_name, visibility_timeout):
    """Fetch task from queue."""
    # This operator is actually a generator,
    # it replaces old tasks to a completely new tasks and loop over it!
    queue = SQSQueue(queue_name, visibility_timeout=visibility_timeout)
    task = INITIAL_TASK
    task['queue'] = queue
    for task_handle, output_bbox_str in queue:
        print('get task: ', output_bbox_str)
        output_bbox = Bbox.from_filename(output_bbox_str)
        # record the task handle to delete after the processing
        task['task_handle'] = task_handle
        task['output_bbox'] = output_bbox
        task['log']['output_bbox'] = output_bbox.to_filename()
        yield task


@main.command('create-chunk')
@click.option('--size',
              type=int,
              nargs=3,
              default=(64, 64, 64),
              help='the size of created chunk')
@click.option('--dtype',
              type=click.Choice(
                  ['uint8', 'uint32', 'uint16', 'float32', 'float64']),
              default='uint8',
              help='the data type of chunk')
@click.option('--voxel-offset',
              type=int,
              nargs=3,
              default=(0, 0, 0),
              help='offset in voxel number.')
@click.option('--output-chunk-name',
              '-o',
              type=str,
              default="chunk",
              help="name of created chunk")
@operator
def create_chunk(tasks, size, dtype, voxel_offset, output_chunk_name):
    """Create a fake chunk for easy test."""
    create_chunk_operator = CreateChunkOperator()
    print("creating chunk: ", output_chunk_name)
    for task in tasks:
        task[output_chunk_name] = create_chunk_operator(
            size=size, dtype=dtype, voxel_offset=voxel_offset)
        yield task


@main.command('read-tif')
@click.option('--name',
              type=str,
              default='read-tif',
              help='read tif file from local disk.')
@click.option('--file-name',
              '-f',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help='read chunk from file, support .h5 and .tif')
@click.option('--offset',
              type=int,
              nargs=3,
              callback=default_none,
              help='global offset of this chunk')
@click.option('--output-chunk-name',
              '-o',
              type=str,
              default='chunk',
              help='chunk name in the global state')
@operator
def read_tif(tasks, name: str, file_name: str, offset: tuple,
             output_chunk_name: str):
    """Read tiff files."""
    read_tif_operator = ReadTIFOperator()
    for task in tasks:
        start = time()
        assert output_chunk_name not in task
        task[output_chunk_name] = read_tif_operator(file_name,
                                                    global_offset=offset)
        task['log']['timer'][name] = time() - start
        yield task


@main.command('read-h5')
@click.option('--name',
              type=str,
              default='read-h5',
              help='read file from local disk.')
@click.option('--file-name',
              '-f',
              type=str,
              required=True,
              help='read chunk from file, support .h5 and .tif')
@click.option('--dataset-path',
              '-d',
              type=str,
              default='/main',
              help='the dataset path inside HDF5 file.')
@click.option('--offset',
              type=int,
              nargs=3,
              callback=default_none,
              help='global offset of this chunk')
@click.option('--output-chunk-name', '-o',
              type=str, default='chunk',
              help='chunk name in the global state')
@operator
def read_h5(tasks, name: str, file_name: str, dataset_path: str, offset: tuple,
            output_chunk_name: str):
    """Read HDF5 files."""
    read_h5_operator = ReadH5Operator()
    for task in tasks:
        start = time()
        assert output_chunk_name not in task
        task[output_chunk_name] = read_h5_operator(file_name,
                                                   dataset_path=dataset_path,
                                                   global_offset=offset)
        task['log']['timer'][name] = time() - start
        yield task


@main.command('write-h5')
@click.option('--name', type=str, default='write-h5', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default='chunk', help='input chunk name')
@click.option('--file-name',
              '-f',
              type=click.Path(dir_okay=False, resolve_path=True),
              required=True,
              help='file name of hdf5 file, the extention should be .h5')
@operator
def write_h5(tasks, name, input_chunk_name, file_name):
    """Write chunk to HDF5 file."""
    state['operators'][name] = WriteH5Operator()
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            state['operators'][name](task[input_chunk_name], file_name)
        yield task


@main.command('write-tif')
@click.option('--name', type=str, default='write-tif', help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default='chunk', help='input chunk name')
@click.option(
    '--file-name',
    '-f',
    type=click.Path(dir_okay=False, resolve_path=True),
    required=True,
    help='file name of tif file, the extention should be .tif or .tiff')
@operator
def write_tif(tasks, name, input_chunk_name, file_name):
    """Write chunk as a TIF file."""
    state['operators'][name] = WriteTIFOperator()
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            state['operators'][name](task[input_chunk_name], file_name)
        # keep the pipeline going
        yield task


@main.command('save-pngs')
@click.option('--name', type=str, default='save-pngs', help='name of operator')
@click.option('--input-chunk-name', '-i', type=click.Path(dir_okay=True, resolve_path=True))
@click.option('--output-path',
              '-o',
              type=str,
              default='./saved_pngs/',
              help='output path of saved 2d images formated as png.')
@operator
def save_pngs(tasks, name, input_chunk_name, output_path):
    """Save as 2D PNG images."""
    state['operators'][name] = SavePNGsOperator(output_path=output_path,
                                                name=name)
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            state['operators'][name](task[input_chunk_name])
        yield task


@main.command('delete-task-in-queue')
@click.option('--name',
              type=str,
              default='delete-task-in-queue',
              help='name of this operator')
@operator
def delete_task_in_queue(tasks, name):
    """Delete the task in queue."""
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            queue = task['queue']
            task_handle = task['task_handle']
            queue.delete(task_handle)
            if state['verbose']:
                print('deleted task {} in queue: {}'.format(
                    task_handle, queue))


@main.command('cutout')
@click.option('--name',
              type=str,
              default='cutout',
              help='name of this operator')
@click.option('--volume-path',
              '-v',
              type=str,
              required=True,
              help='volume path')
@click.option('--mip',
              '-m',
              type=int,
              default=None,
              help='mip level of the cutout.')
@click.option('--expand-margin-size',
              '-e',
              type=int,
              nargs=3,
              default=(0, 0, 0),
              help='include surrounding regions of output bounding box.')
@click.option('--start',
              '-s',
              type=int,
              nargs=3,
              default=None,
              help='chunk offset in volume.')
@click.option('--stop',
              '-p',
              type=int,
              nargs=3,
              default=None,
              help='chunk stop coordinate.')
@click.option('--fill-missing/--no-fill-missing',
              default=False,
              help='fill the missing chunks in input volume with zeros ' +
              'or not, default is false')
@click.option('--validate-mip',
              type=int,
              default=None,
              help='validate chunk using higher mip level')
@click.option(
    '--blackout-sections/--no-blackout-sections',
    default=False,
    help='blackout some sections. ' +
    'the section ids json file should named blackout_section_ids.json.' +
    'default is False.')
@click.option(
    '--output-chunk-name',
    '-o',
    type=str,
    default='chunk',
    help='Variable name to store the cutout to for later retrieval. Chunkflow'
    ' operators by default operates on a variable named "chunk" but'
    ' sometimes you may need to have a secondary volume to work on.')
@operator
def cutout(tasks, name, volume_path, mip, start, stop, expand_margin_size,
           fill_missing, validate_mip, blackout_sections, output_chunk_name):
    """Cutout chunk from volume."""
    if mip is None:
        mip = state['mip']
    state['operators'][name] = CutoutOperator(
        volume_path,
        mip=mip,
        expand_margin_size=expand_margin_size,
        verbose=state['verbose'],
        fill_missing=fill_missing,
        validate_mip=validate_mip,
        blackout_sections=blackout_sections,
        name=name)

    for task in tasks:
        handle_task_skip(task, name)
        if start is None and stop is None:
            bbox = task['output_bbox']
        else:
            # use bounding box of volume
            if stop is None:
                stop = state['operators'][name].vol.mip_bounds(mip).maxpt
            if start is None:
                start = state['operators'][name].vol.mip_bounds(mip).minpt
            bbox = Bbox(start, stop)

        if not task['skip']:
            start = time()
            assert output_chunk_name not in task
            task[output_chunk_name] = state['operators'][name](bbox)
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
    """compute split/merge error using rand index
     and variation of information.
    """
    for task in tasks:
        seg = Segmentation(task[segmentation_chunk_name])
        groundtruth = Segmentation(task[groundtruth_chunk_name])
        seg.evaluate(groundtruth)
        yield task


@main.command('downsample-upload')
@click.option('--name',
              type=str,
              default='downsample-upload',
              help='name of operator')
@click.option('--input-chunk-name', '-i',
              type=str, default='chunk', help='input chunk name')
@click.option('--volume-path', type=str, help='path of output volume')
@click.option('--start-mip',
              type=int,
              default=0,
              help='the start uploading mip level.')
@click.option(
    '--stop-mip',
    type=int,
    default=5,
    help='stop mip level. the indexing follows python style and ' +
    'the last index is exclusive.')
@click.option('--fill-missing/--no-fill-missing',
              default=True,
              help='fill missing or not when there is all zero blocks.')
@operator
def downsample_upload(tasks, name, input_chunk_name, volume_path, 
                      start_mip, stop_mip, fill_missing):
    """Downsample chunk and upload to volume."""
    state['operators'][name] = DownsampleUploadOperator(
        volume_path,
        input_mip=state['mip'],
        start_mip=start_mip,
        stop_mip=stop_mip,
        fill_missing=fill_missing,
        name=name)

    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            state['operators'][name](task[input_chunk_name])
            task['log']['timer'][name] = time() - start
        yield task


@main.command('normalize-section-contrast')
@click.option('--name',
              type=str,
              default='normalize-section-contrast',
              help='name of operator.')
@click.option('--levels-path',
              type=str,
              default=None,
              help='the path of section histograms.')
@click.option('--mip',
              type=int,
              default=None,
              help='the mip level of section histograms.')
@click.option('--clip-fraction',
              type=float,
              default=0.01,
              help='the voxel intensity fraction to clip out.')
@click.option('--minval',
              type=float,
              default=None,
              help='the minimum intensity of transformed chunk.')
@click.option('--maxval',
              type=float,
              default=None,
              help='the maximum intensity of transformed chunk.')
@operator
def normalize_contrast_contrast(tasks, name, levels_path, mip, clip_fraction,
                                minval, maxval):
    """Normalize the section contrast using precomputed histograms."""
    if mip is None:
        mip = state['mip']
    if levels_path is None:
        levels_path = state['cutout_volume_path']

    state['operators'][name] = NormalizeSectionContrastOperator(levels_path,
                                                                mip,
                                                                clip_fraction,
                                                                minval=minval,
                                                                maxval=maxval,
                                                                name=name)

    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            task['chunk'] = state['operators'][name](task['chunk'])
            task['log']['timer'][name] = time() - start
        yield task


@main.command('normalize-section-shang')
@click.option('--name',
              type=str,
              default='normalize-section-mu',
              help='name of operator.')
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
def normalize_section_shang(tasks, name, nominalmin, nominalmax, clipvalues):
    """Normalize voxel values based on slice min/max within the chunk, Shang's method.
    The transformed chunk has floating point values.
    """

    state['operators'][name] = NormalizeSectionShangOperator(
        nominalmin=nominalmin,
        nominalmax=nominalmax,
        clipvalues=clipvalues,
        name=name)

    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            task['chunk'] = state['operators'][name](task['chunk'])
            task['log']['timer'][name] = time() - start
        yield task


@main.command('custom-operator')
@click.option('--name',
              type=str,
              default='custom-operator-1',
              help='name of operator.')
@click.option('--input-chunk-name', '-i',
              type=str, default='chunk', help='input chunk name')
@click.option('--output-chunk-name', '-o',
              type=str, default='chunk', help='output chunk name')
@click.option('--opprogram', type=str, help='python file to call.')
@click.option('--args', type=str, default='', help='args to pass in')
@operator
def custom_operator(tasks, name, input_chunk_name, output_chunk_name, opprogram, args):
    """Custom operation on the chunk.
    The custom python file should contain a callable named "op_call" such that 
    a call of `op_call(chunk, args)` can be made to operate on the chunk.
    """

    state['operators'][name] = CustomOperator(opprogram=opprogram,
                                              args=args,
                                              name=name)
    if state['verbose']:
        print('Received args for ', name, ':', args)

    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            task[output_chunk_name] = state['operators'][name](task[input_chunk_name])
            task['log']['timer'][name] = time() - start
        yield task


@main.command('connected-components')
@click.option('--name', type=str, default='connected-components', 
              help='threshold a map and get the labels.')
@click.option('--input-chunk-name', '-i',
              type=str, default='chunk', help='input chunk name')
@click.option('--output-chunk-name', '-o',
              type=str, default='chunk', help='output chunk name')
@click.option('--threshold', '-t', type=float, default=0.5,
              help='threshold to cut the map.')
@click.option('--connectivity', '-c', 
              type=click.Choice([6, 18, 26]),
              default=26, help='number of neighboring voxels used.')
@operator 
def connected_components(tasks, name, input_chunk_name, output_chunk_name, 
                         threshold, connectivity):
    """Threshold the map to get a segmentation."""
    state['operators'][name] = ConnectedComponentsOperator(name=name, 
                                                           threshold=threshold, 
                                                           connectivity=connectivity)
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            task[output_chunk_name] = state['operators'][name](task[input_chunk_name])
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
@click.option('--name',
              type=str,
              default='inference',
              help='name of this operator')
@click.option('--convnet-model', '-m',
              type=str,
              default=None,
              help='convnet model path or type.')
@click.option('--convnet-weight-path', '-w',
              type=str,
              default=None,
              help='convnet weight path')
@click.option('--patch-size', '-s',
              type=int,
              nargs=3,
              default=(20, 256, 256),
              help='patch size')
@click.option('--patch-overlap', '-v',
              type=int,
              nargs=3,
              default=(4, 64, 64),
              help='patch overlap')
@click.option('--output-key',
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
@click.option('--num-output-channels', '-c',
              type=int,
              default=3,
              help='number of output channels')
@click.option('--framework', '-f',
              type=click.Choice(
                  ['identity', 'pznet', 'pytorch', 'pytorch-multitask']),
              default='pytorch-multitask',
              help='inference framework')
@click.option('--batch-size', '-b',
              type=int,
              default=1,
              help='mini batch size of input patch.')
@click.option(
    '--bump',
    type=click.Choice(['wu', 'zung']),
    default='wu',
    help='bump function type. only works with pytorch-multitask backend.')
@click.option(
    '--mask-output-chunk/--no-mask-output-chunk',
    default=False,
    help='mask output chunk will make the whole chunk like one output patch. '
    + 'This will also work with non-aligned chunk size.')
@click.option(
    '--input-chunk-name', '-i',
    type=str,
    default='chunk',
    help='input chunk name')
@click.option(
    '--output-chunk-name', '-o',
    type=str,
    default='chunk',
    help='output chunk name')
@operator
def inference(tasks, name, convnet_model, convnet_weight_path, patch_size,
              patch_overlap, output_key, original_num_output_channels,
              num_output_channels, framework, batch_size, bump,
              mask_output_chunk, input_chunk_name, output_chunk_name):
    """Perform convolutional network inference for chunks."""
    state['operators'][name] = InferenceOperator(
        convnet_model,
        convnet_weight_path,
        patch_size=patch_size,
        output_key=output_key,
        num_output_channels=num_output_channels,
        original_num_output_channels=original_num_output_channels,
        patch_overlap=patch_overlap,
        framework=framework,
        batch_size=batch_size,
        bump=bump,
        mask_output_chunk=mask_output_chunk,
        verbose=state['verbose'],
        name=name)

    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            if 'log' not in task:
                task['log'] = {'timer': {}}
            start = time()

            task[output_chunk_name] = state['operators'][name](
                task[input_chunk_name])

            task['log']['timer'][name] = time() - start
            task['log']['compute_device'] = state['operators'][
                name].compute_device
        yield task


@main.command('mask')
@click.option('--name', type=str, default='mask', help='name of this operator')
@click.option('--volume-path',
              type=str,
              required=True,
              help='mask volume path')
@click.option('--mip', type=int, default=5, help='mip level of mask')
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
@click.option('--skip-to', type=str, default='save', help='skip to a operator')
@operator
def mask(tasks, name, volume_path, mip, inverse, fill_missing, check_all_zero,
         skip_to):
    """Mask the chunk. The mask could be in higher mip level and we
    will automatically upsample it to the same mip level with chunk.
    """
    state['operators'][name] = MaskOperator(volume_path,
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
                task['skip'] = state['operators'][name].is_all_zero(
                    task['output_bbox'])
                if task['skip']:
                    print('the mask of {} is all zero, will skip to {}'.format(
                        name, skip_to))
                task['skip_to'] = skip_to
            else:
                task['chunk'] = state['operators'][name](task['chunk'])
            # Note that mask operation could be used several times,
            # this will only record the last masking operation
            task['log']['timer'][name] = time() - start
        yield task


@main.command('crop-margin')
@click.option('--name',
              type=str,
              default='crop-margin',
              help='name of this operator')
@click.option('--margin-size', '-m',
              type=int,
              nargs=3,
              default=None,
              help='crop the chunk margin. ' +
              'The default is None and will use the output_bbox ' +
              'as croping range.')
@click.option('--chunk-name', '-c',
              type=str,
              default='chunk',
              help='input and output chunk name.')
@operator
def crop_margin(tasks, name, margin_size, chunk_name):
    """Crop the margin of chunk."""
    state['operators'][name] = CropMarginOperator(margin_size=margin_size,
                                                  verbose=state['verbose'],
                                                  name=name)
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            task[chunk_name] = state['operators'][name](
                task[chunk_name], output_bbox=task.get('output_bbox', None))
            task['log']['timer'][name] = time() - start
        yield task


@main.command('mesh')
@click.option('--name', type=str, default='mesh', help='name of operator')
@click.option('--chunk-name',
              type=str,
              default='chunk',
              help='name of chunk needs to be meshed.')
@click.option('--voxel-size',
              type=int,
              nargs=3,
              default=(40, 4, 4),
              help='voxel size of the segmentation. zyx order.')
@click.option(
    '--output-path',
    type=str,
    default='file:///tmp/mesh/',
    help='output path of meshes, follow the protocol rule of CloudVolume. \
              The path will be adjusted if there is a info file with precomputed format.'
)
@click.option('--output-format',
              type=click.Choice(['ply', 'obj', 'precomputed']),
              default='ply',
              help='output format, could be one of ply|obj|precomputed.')
@click.option('--simplification-factor',
              type=int,
              default=100,
              help='mesh simplification factor.')
@click.option('--max-simplification-error',
              type=int,
              default=8,
              help='max simplification error.')
@click.option('--manifest/--no-manifest',
              default=False,
              help='create manifest file or not.')
@operator
def mesh(tasks, name, chunk_name, voxel_size, output_path, output_format,
         simplification_factor, max_simplification_error, manifest):
    """Perform meshing for segmentation chunk."""
    state['operators'][name] = MeshOperator(
        output_path,
        output_format,
        voxel_size=voxel_size,
        simplification_factor=simplification_factor,
        max_simplification_error=max_simplification_error,
        manifest=manifest)
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            start = time()
            task[chunk_name] = state['operators'][name](task[chunk_name])
            task['log']['timer'][name] = time() - start
        yield task


@main.command('save')
@click.option('--name', type=str, default='save', help='name of this operator')
@click.option('--volume-path', type=str, required=True, help='volume path')
@click.option('--upload-log/--no-upload-log',
              default=True,
              help='the log will be put inside volume-path')
@click.option(
    '--nproc',
    type=int,
    default=0,
    help='number of processes, negative means using all the cores, ' +
    '0/1 means turning off multiple processing, ' +
    'n>1 means using n processes')
@click.option(
    '--create-thumbnail/--no-create-thumbnail',
    default=False,
    help='create thumbnail or not. ' +
    'the thumbnail is a downsampled and quantized version of the chunk.')
@operator
def save(tasks, name, volume_path, upload_log, nproc, create_thumbnail):
    """Save chunk to volume."""
    state['operators'][name] = SaveOperator(volume_path,
                                            state['mip'],
                                            upload_log=upload_log,
                                            create_thumbnail=create_thumbnail,
                                            nproc=nproc,
                                            verbose=state['verbose'],
                                            name=name)

    for task in tasks:
        # we got a special case for handling skip
        if task['skip'] and task['skip_to'] == name:
            task['skip'] = False
            # create fake chunk to save
            task['chunk'] = state['operators'][name].create_chunk_with_zeros(
                task['output_bbox'])

        if not task['skip']:
            # the time elapsed was recorded internally
            state['operators'][name](task['chunk'],
                                     log=task.get('log', {'timer': {}}),
                                     output_bbox=task.get('output_bbox', None))
            task['output_volume_path'] = volume_path
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
    state['operators'][name] = CloudWatchOperator(log_name=log_name,
                                                  name=name,
                                                  verbose=state['verbose'])
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            state['operators'][name](task['log'])
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
    state['operators'][name] = ViewOperator(name=name)
    for task in tasks:
        handle_task_skip(task, name)
        if not task['skip']:
            state['operators'][name](task[image_chunk_name],
                                     seg=segmentation_chunk_name)
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
    state['operators'][name] = NeuroglancerOperator(name=name,
                                                    port=port,
                                                    voxel_size=voxel_size)
    for task in tasks:
        chunks = dict()
        for chunk_name in chunk_names.split(","):
            chunks[chunk_name] = task[chunk_name]

        handle_task_skip(task, name)
        if not task['skip']:
            state['operators'][name](chunks)
        yield task


if __name__ == '__main__':
    main()
