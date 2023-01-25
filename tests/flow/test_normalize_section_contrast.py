#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(0)

from chunkflow.chunk.image import Image


def test_normalize_section_contrast():
    #histogram = np.random.randint(10000, size=256, dtype=np.uint32)
    image = np.arange(256, dtype=np.uint8).reshape(1, 16, 16)
    assert image.ndim == 3
    image = Image(image, voxel_offset=(26774, 234, 456))
    
    normalized = image.normalize_contrast()
