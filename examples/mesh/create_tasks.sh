#!/bin/bash

chunkflow generate-tasks -l gs://my/dataset/path -m 3 -o 1 1 1 -c 226 2050 2050 -q my-queue
