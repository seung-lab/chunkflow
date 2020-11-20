#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import logging
import os

import numpy as np

from cloudvolume.lib import Vec, yellow
from cloudvolume.storage import SimpleStorage
from cloudvolume import CloudVolume

from chunkflow.lib.bounding_boxes import BoundingBoxes


def tuple2string(tp: tuple):
    return ' '.join(str(i) for i in tp)


def get_optimized_block_size(
        output_patch_size, output_patch_overlap, max_ram_size,
        channel_num, max_mip, crop_chunk_margin,
        input_patch_size, mip, thumbnail_mip, volume_start):
    patch_stride = tuple(s - o for s, o in zip(
    output_patch_size, output_patch_overlap))
    # total number of voxels per patch in one stride
    patch_voxel_num = np.product(patch_stride)
    # use half of the maximum ram size to store output buffer
    ideal_total_patch_num = int(max_ram_size*1e9/2/4/channel_num/patch_voxel_num)
    # the xy size should be the same
    assert output_patch_size[1] == output_patch_size[2]
    # compute the output chunk/block size in cloud storage
    # assume that the crop margin size is the same with the patch overlap
    patch_num_start = int(ideal_total_patch_num ** (1./3.) / 2)
    patch_num_stop = patch_num_start * 3

    # find the patch number solution with minimum cost by bruteforce search
    cost = sys.float_info.max
    patch_num = None
    # patch number in x and y
    max_factor = 2**max_mip
    factor = 2**mip
    for pnxy in range(patch_num_start, patch_num_stop):
        if (pnxy * patch_stride[2] + output_patch_overlap[2] -
                2 * crop_chunk_margin[2]) % max_factor != 0:
            continue
        # patch number in z
        for pnz in range(patch_num_start, patch_num_stop):
            if (pnz * patch_stride[0] + output_patch_overlap[0] -
                    2 * crop_chunk_margin[0]) % factor != 0:
                continue
            current_cost = ( pnxy * pnxy * pnz / ideal_total_patch_num - 1) ** 2 #+ (pnxy / pnz - 1) ** 2
            if current_cost < cost:
                cost = current_cost
                patch_num = (pnz, pnxy, pnxy)
    
    print('\n--input-patch-size ', tuple2string(input_patch_size))
    print('--output-patch-size ', tuple2string(output_patch_size))
    print('--output-patch-overlap ', tuple2string(output_patch_overlap))
    print('--output-patch-stride ', tuple2string(patch_stride))
    print('--patch-num ', patch_num)

    assert mip>=0
    block_mip = (mip + thumbnail_mip) // 2
    block_factor = 2 ** block_mip
    
    output_chunk_size = tuple(n*s+o-2*c for n, s, o, c in zip(
        patch_num, patch_stride, output_patch_overlap, crop_chunk_margin))
    
    input_chunk_size = tuple(ocs + ccm*2 + ips - ops for ocs, ccm, ips, ops in zip(
        output_chunk_size, crop_chunk_margin,
        input_patch_size, output_patch_size))
    
    expand_margin_size = tuple((ics - ocs) // 2 for ics, ocs in zip(
        input_chunk_size, output_chunk_size))

    input_chunk_start = tuple(vs - ccm - (ips-ops)//2 for vs, ccm, ips, ops in zip(
        volume_start, crop_chunk_margin, input_patch_size, output_patch_size))
    
    block_size = (output_chunk_size[0]//factor,
                output_chunk_size[1]//block_factor,
                output_chunk_size[2]//block_factor)

    print('\n--input-chunk-size ' + tuple2string(input_chunk_size))
    print('--input-volume-start ' + tuple2string(input_chunk_start))
    print('--output-chunk-size ' + tuple2string(output_chunk_size))
    print('cutout expand margin size ' + tuple2string(expand_margin_size))

    print('output volume start: ' + tuple2string(volume_start))
    print('block size ' + tuple2string(block_size))
    print('size of each block (uncompressed, uint8, 1 channel): ', 
            np.prod(block_size)/1e6, ' MB')
    print('RAM size of each block: ',
            np.prod(output_chunk_size)/1024/1024/1024*4*channel_num, ' GB')
    voxel_utilization = np.prod(output_chunk_size)/np.prod(patch_num)/np.prod(output_patch_size)
    print('voxel utilization: {:.2f}'.format(voxel_utilization))

    return block_size, output_chunk_size, factor


def setup_environment(dry_run, volume_start, volume_stop, volume_size, layer_path, 
              max_ram_size, output_patch_size, 
              input_patch_size, channel_num, dtype, 
              output_patch_overlap, crop_chunk_margin, mip, thumbnail_mip, max_mip,
              thumbnail, encoding, voxel_size, 
              overwrite_info):
    """Prepare storage info files and produce tasks."""
    assert not (volume_stop is None and volume_size is None)
    if isinstance(volume_start, tuple):
        volume_start = Vec(*volume_start)
    if isinstance(volume_stop, tuple):
        volume_stop = Vec(*volume_stop)
    if isinstance(volume_size, tuple):
        volume_size = Vec(*volume_size)

    if input_patch_size is None:
        input_patch_size = output_patch_size

    if volume_size is not None:
        assert len(volume_size) == 3
        assert volume_stop is None
        volume_stop = volume_start + volume_size
    else:
        volume_size = volume_stop - volume_start
    print('\noutput volume start: ' + tuple2string(volume_start))
    print('output volume stop: ' + tuple2string(volume_stop))
    print('output volume size: ' + tuple2string(volume_size))
    
    if output_patch_overlap is None:
        # use 50% patch overlap in default
        output_patch_overlap = tuple(s//2 for s in output_patch_size)
    assert output_patch_overlap[1] == output_patch_overlap[2]

    if crop_chunk_margin is None:
        crop_chunk_margin = output_patch_overlap
    assert crop_chunk_margin[1] == crop_chunk_margin[2]
    print('margin size: ' + tuple2string(crop_chunk_margin))
    
    if thumbnail:
        # thumnail requires maximum mip level of 5
        thumbnail_mip = max(thumbnail_mip, 5)

    block_size, output_chunk_size, factor = get_optimized_block_size(
        output_patch_size, output_patch_overlap, max_ram_size,
        channel_num, max_mip, crop_chunk_margin,
        input_patch_size, mip, thumbnail_mip, volume_start
    )

    if not dry_run:
        storage = SimpleStorage(layer_path)
        thumbnail_layer_path = os.path.join(layer_path, 'thumbnail')
        thumbnail_storage = SimpleStorage(thumbnail_layer_path)

        if not overwrite_info:
            print('\ncheck that we are not overwriting existing info file.')
            assert storage.exists('info')
            assert thumbnail_storage.exists('info')

        if overwrite_info:
            print('create and upload info file to ', layer_path)
            # Note that cloudvolume use fortran order rather than C order
            info = CloudVolume.create_new_info(channel_num, layer_type='image',
                                            data_type=dtype,
                                            encoding=encoding,
                                            resolution=voxel_size[::-1],
                                            voxel_offset=volume_start[::-1],
                                            volume_size=volume_size[::-1],
                                            chunk_size=block_size[::-1],
                                            max_mip=mip)
            vol = CloudVolume(layer_path, info=info)
            vol.commit_info()
      
        if overwrite_info:
            thumbnail_factor = 2**thumbnail_mip
            thumbnail_block_size = (output_chunk_size[0]//factor,
                                    output_chunk_size[1]//thumbnail_factor,
                                    output_chunk_size[2]//thumbnail_factor)
            print('thumbnail block size: ' + tuple2string(thumbnail_block_size))
            thumbnail_info = CloudVolume.create_new_info(
                1, layer_type='image', 
                data_type='uint8',
                encoding='raw',
                resolution=voxel_size[::-1],
                voxel_offset=volume_start[::-1],
                volume_size=volume_size[::-1],
                chunk_size=thumbnail_block_size[::-1],
                max_mip=thumbnail_mip)
            thumbnail_vol = CloudVolume(thumbnail_layer_path, info=thumbnail_info)
            thumbnail_vol.commit_info()
       
    print('create a list of bounding boxes...')
    roi_start = (volume_start[0],
                 volume_start[1]//factor,
                 volume_start[2]//factor)
    roi_size = (volume_size[0],
                volume_size[1]//factor,
                volume_size[2]//factor)
    roi_stop = tuple(s+z for s, z in zip(roi_start, roi_size))

    # create bounding boxes and ingest to queue
    bboxes = BoundingBoxes.from_manual_setup(
            output_chunk_size,
            roi_start=roi_start, roi_stop=roi_stop)
    logging.info(f'total number of tasks: {len(bboxes)}')
    
    logging.debug(f'bounding boxes: {bboxes}')
    
    print(yellow(
        'Note that you should reuse the printed out parameters in the production run.' + 
        ' These parameters are not ingested to AWS SQS queue.'))
    return bboxes


