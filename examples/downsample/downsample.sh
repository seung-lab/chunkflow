#!/bin/bash
export PROCESS_NUM=16
export MIP=0
export STOP_MIP=6
export DATASET_PATH="gs://my/dataset/path"

# perform downsample
seq "$PROCESS_NUM" | parallel -j "$PROCESS_NUM" --delay 1 --ungroup echo Starting worker {}\; chunkflow --quiet --mip "$MIP" fetch-task -q my-queue-name -v 60 cutout -v "$DATASET_PATH" --fill-missing downsample-upload -v "$DATASET_PATH" --stop-mip "$STOP_MIP" delete-task-in-queue
