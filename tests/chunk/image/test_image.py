#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from chunkflow.chunk.image import Image

def test_image_construction():
    arr = np.random.randint(256, size=(3,4,5), dtype=np.uint8)
    image = Image(arr, global_offset=(-1,-1,-1))
 
