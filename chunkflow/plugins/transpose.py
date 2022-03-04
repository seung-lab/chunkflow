#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chunkflow.chunk import Chunk


def execute(chunk: Chunk):
    chunk = chunk.transpose()
    return [chunk,]
