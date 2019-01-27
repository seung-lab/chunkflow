import click 
import numpy as np
from cloudvolume import CloudVolume, Bbox 
from chunkflow.sqs_queue import SQSQueue 
from chunkflow.executor import Executor

@click.command()
@click.option('--image-layer-path', type=str, required=True, help='image layer path')
@click.option('--output-layer-path', type=str, required=True, help='output layer path')
@click.option('--convnet-model-path', type=str, required=True, help='convnet model path')
@click.option('--convnet-weight-path', type=str, required=True, help='convnet weight path')
@click.option('--image-mask-layer-path', type=str, default=None, help='image mask layer path')
@click.option('--output-mask-layer-path', type=str, default=None, help='output mask layer path')
@click.option('--output-offset', type=int, nargs=3, default=(0,0,0), help='output offset')
@click.option('--output-shape', type=int, nargs=3, default=(0,0,0), help='output shape')
@click.option('--queue-name', type=str, default=None, help='sqs queue name')
@click.option('--patch-size', type=int, nargs=3, required=True, help='patch size')
@click.option('--patch-overlap', type=int, nargs=3, default=(4,64,64), help='patch overlap')
@click.option('--cropping-margin-size', type=int, nargs=3, default=(4,64,64), help='cropping size')
@click.option('--output-key', type=str, default='affinity', help='key name of output dict')
@click.option('--num-output-channels', type=int, default=3, help='number of output channels')
@click.option('--mip', type=int, default=1, help='mip level of image and output')
@click.option('--output-mask-mip', type=int, default=None, help='mip level of output mask')
@click.option('--framework', type=click.Choice(['pznet', 'pytorch', 'pytorch-multitask']), 
              default='pytorch-multitask', help='inference framework')
@click.option('--missing-section_ids_file_name', type=str, default=None, 
              help='black out the missing sections recorded in a txt file.' +\ 
              'the section id is simply a list of z coordinates of missing sections')
@click.option('--image-validate-mip', type=int, default=5, help='validate image using mip level')
@click.option('--visibility-timeout', type=int, default=1800, help='visibility timeout of sqs queue')


def command(image_layer_path, output_layer_path, convnet_model_path, convnet_weight_path, 
            image_mask_layer_path, output_mask_layer_path, output_offset, output_shape, queue_name, 
            patch_size, patch_overlap, cropping_margin_size, output_key, num_output_channels, mip, 
            output_mask_mip, framework, missing_section_ids_file_name, image_validate_mip, 
            visibility_timeout):
    executor = Executor(image_layer_path, output_layer_path, convnet_model_path, convnet_weight_path, 
                        image_mask_layer_path, output_mask_layer_path, queue_name, patch_size, 
                        patch_overlap, cropping_margin_size, output_key, num_output_channels, mip, 
                        output_mask_mip, framework, missing_section_ids_file_name, image_validate_mip) 
    if queue_name: 
        # read from sqs queue 
        sqs_queue = SQSQueue(queue_name, visibility_timeout=visibility_timeout)
        for task in sqs_queue:
            output_bbox = Bbox.from_filename(task)
            executor(output_bbox)
    else:
        output_stop = np.asarray(output_offset) + np.asarray(output_shape)
        output_bbox = Bbox.from_list([*output_offset, *output_stop])
        executor(output_bbox)


if __name__ == '__main__':
    command()
