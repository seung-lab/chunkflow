#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil
import tempfile
from time import sleep

import numpy as np
from cloudvolume import CloudVolume

from chunkflow.chunk import Chunk
from chunkflow.flow.save_precomputed import SavePrecomputedOperator



def test_save_image():
    mip = 0
    size = (8, 64, 64)
    voxel_offset = (2, 4, 3)

    chunk = Chunk.create(size=size, dtype=np.uint8, 
                         voxel_offset=voxel_offset) 

    tempdir = tempfile.mkdtemp()
    volume_path = 'file://' + tempdir
    print('construct volume from numpy array in ', tempdir)
    vol = CloudVolume.from_numpy(
        chunk.transpose(),
        vol_path=volume_path,
        voxel_offset=voxel_offset[::-1],
        chunk_size=(32, 32, 4),
        max_mip=1,
        layer_type='image'
    )

    print('construct save operator')
    op = SavePrecomputedOperator(
        volume_path, mip,
        upload_log = True,
        create_thumbnail = False,
        name = 'save'
    )
    
    print('really save the chunk.')
    op(chunk, log={'timer': {'save': 43}})
    
    sleep(2)
    shutil.rmtree(tempdir)
