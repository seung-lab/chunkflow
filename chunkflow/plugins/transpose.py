#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chunkflow.chunk import Chunk


def execute(chunk: Chunk):
    chunk.transpose()
    return [chunk,]
