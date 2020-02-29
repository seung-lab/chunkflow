#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tempfile

from chunkflow.chunk import Chunk
from chunkflow.flow.mesh import MeshOperator


def test_mesh():
    ck = Chunk.create(dtype=np.float32)
    # segment it with threshold 
    ck = ck < 0.5
    ck = ck.astype(np.uint32)
    
    tempdir = tempfile.mkdtemp()
    #mesher = MeshOperator('file://' + tempdir, 'precomputed', manifest=True)
    mesher = MeshOperator('file://' + tempdir, 'obj')
    mesher(ck)

    mesher = MeshOperator('file://' + tempdir, 'ply')
    mesher(ck)
