#!/bin/bash

# downsample from mip 0 to mip 5
chunkflow generate-tasks -l gs://my/dataset/path -m 0 -o 0 0 0 -c 112 2048 2048 -q my-queue-name

# downsample from mip 5 to mip 9
#chunkflow generate-tasks -l gs://my/dataset/path -m 5 -o 0 0 0 -c 112 2048 2048 -q my-queue-name
