#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chunkflow.chunk import Chunk


def execute(chunk: Chunk, only_array: bool=True):
    chunk = chunk.transpose(only_array=only_array)
    return [chunk,]
