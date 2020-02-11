#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
np.random.seed(0)

from chunkflow.chunk import Chunk
from chunkflow.flow.normalize_section_contrast import NormalizeSectionContrastOperator


def test_normalize_section_contrast():
    DIR = os.path.join(os.path.dirname(__file__), '../data/levels/1/')
    #histogram = np.random.randint(10000, size=256, dtype=np.uint32)
    image = np.arange(256, dtype=np.uint8).reshape(1, 16, 16)
    image = Chunk(image, global_offset=(26774, 234, 456))
    
    levels_path = 'precomputed://file://' + DIR
    operator = NormalizeSectionContrastOperator(levels_path)
    normalized = operator(image)
    
    normalized_filename = os.path.join(DIR, '../normalized.npz')
    #np.savez( normalized_filename , normalized=normalized)
    groundtruth = np.load(normalized_filename)['normalized']
    np.testing.assert_array_equal(normalized, groundtruth)
