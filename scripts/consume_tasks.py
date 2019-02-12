import click 
import numpy as np
from cloudvolume import CloudVolume, Bbox 
from chunkflow.sqs_queue import SQSQueue 
from chunkflow.executor import Executor
import multiprocessing as mp
import time 
import traceback 
from cloudvolume import EmptyVolumeException

@click.command()
@click.option('--image-layer-path', type=str, required=True, help='image layer path')
@click.option('--output-layer-path', type=str, required=True, help='output layer path')
@click.option('--convnet-model-path', type=str, required=True, help='convnet model path')
@click.option('--convnet-weight-path', type=str, required=True, help='convnet weight path')
@click.option('--output-offset', type=int, nargs=3, default=(0,0,0), help='output offset')
@click.option('--output-shape', type=int, nargs=3, default=(0,0,0), help='output shape')
@click.option('--queue-name', type=str, default=None, help='sqs queue name')
@click.option('--patch-size', type=int, nargs=3, required=True, help='patch size')
@click.option('--patch-overlap', type=int, nargs=3, default=(4,64,64), help='patch overlap')
@click.option('--cropping-margin-size', type=int, nargs=3, default=(4,64,64), help='cropping size')
@click.option('--output-key', type=str, default='affinity', help='key name of output dict')
@click.option('--num-output-channels', type=int, default=3, help='number of output channels')
@click.option('--mip', type=int, default=1, help='mip level of image and output')
@click.option('--image-mask-layer-path', type=str, default=None, help='image mask layer path')
@click.option('--output-mask-layer-path', type=str, default=None, help='output mask layer path')
@click.option('--image-mask-mip', type=int, default=None, help='mip level of output mask')
@click.option('--output-mask-mip', type=int, default=None, help='mip level of output mask')
@click.option('--inverse-image-mask/--not-inverse-image-mask', is_flag=True, help='inverse the image mask or not.')
@click.option('--inverse-output-mask/--not-inverse-output-mask', is_flag=True, help='inverse the output mask or not.')
@click.option('--framework', type=click.Choice(['pznet', 'pytorch', 'pytorch-multitask']), 
              default='pytorch-multitask', help='inference framework')
@click.option('--missing-section_ids_file_name', type=str, default=None, 
              help='black out the missing sections recorded in a txt file.' +  
              'the section id is simply a list of z coordinates of missing sections')
@click.option('--image-validate-mip', type=int, default=5, help='validate image using mip level')
@click.option('--visibility-timeout', type=int, default=None, help='visibility timeout of sqs queue')
@click.option('--proc-num', type=int, default=1, 
              help='number of processes. if set <=0, will equal to the number of cores.')
@click.option('--interval', type=int, default=0, help='interval of processes start time (sec)')


def command(image_layer_path, output_layer_path, convnet_model_path, convnet_weight_path, 
            output_offset, output_shape, queue_name, 
            patch_size, patch_overlap, cropping_margin_size, output_key, num_output_channels, mip, 
            image_mask_layer_path, output_mask_layer_path, image_mask_mip, output_mask_mip,
            inverse_image_mask, inverse_output_mask,
            framework, missing_section_ids_file_name, image_validate_mip, 
            visibility_timeout, proc_num, interval):
    executor = Executor(image_layer_path, output_layer_path, convnet_model_path, convnet_weight_path, 
                        patch_size, 
                        patch_overlap, cropping_margin_size, output_key=output_key, 
                        num_output_channels=num_output_channels, mip=mip, 
                        image_mask_layer_path=image_mask_layer_path, 
                        output_mask_layer_path=output_mask_layer_path, 
                        image_mask_mip=image_mask_mip, output_mask_mip=output_mask_mip,
                        inverse_image_mask=inverse_image_mask, inverse_output_mask=inverse_output_mask,
                        framework=framework, 
                        missing_section_ids_file_name=missing_section_ids_file_name, 
                        image_validate_mip=image_validate_mip) 
    if not queue_name:
        # no queue name specified
        # will only run one task
        output_stop = np.asarray(output_offset) + np.asarray(output_shape)
        output_bbox = Bbox.from_list([*output_offset, *output_stop])
        executor(output_bbox)
    else:
        if proc_num <= 0:
            # use all the cores!
            proc_num = mp.cpu_count()

        if proc_num == 1:
            process_queue(executor, queue_name, visibility_timeout=visibility_timeout)
        else:
            print('launching {} processes.'.format(proc_num))
            with mp.Pool(proc_num) as pool:
                try:
                    for i in range(proc_num):
                        time.sleep(i*interval)
                        print('starting process {}'.format(i))
                        pool.apply_async(process_queue, args=(executor, queue_name),
                                         kwds={'visibility_timeout': visibility_timeout})
                        # this function was used for debugging
                        # pool.apply(process_queue, args=(executor, queue_name, 
                        #                                visibility_timeout))
                    pool.close()
                    pool.join()
                except KeyboardInterrupt:
                    print('Interrupted. Exiting.')
                    pool.terminate()
                    pool.join()

def process_queue(executor, queue_name, visibility_timeout=None):
    assert isinstance(executor, Executor)
    # queue name was defined, read from sqs queue 
    queue = SQSQueue(queue_name, visibility_timeout=visibility_timeout)

    for task_handle, task in queue:
        print('get task: ', task)
        output_bbox = Bbox.from_filename(task)
        try:
            executor(output_bbox)
        except EmptyVolumeException:
            print("raised an EmptyVolumeException, please check the bounds of volume.")
            raise
        except Exception as err:
            print(task, ' raised {}\n {}'.format(err, traceback.format_exc()))
            raise
        queue.delete(task_handle)


if __name__ == '__main__':
    command()
