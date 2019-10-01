#!/bin/bash
export PROCESS_NUM=20
export DATASET_PATH="gs://my/dataset/path"

seq "$PROCESS_NUM" | parallel -j "$PROCESS_NUM" --delay 120 --ungroup echo Starting worker {}\; chunkflow --mip 3 fetch-task -q my-queue -v 1200 cutout -v "$DATASET_PATH" --fill-missing mesh -o "$DATASET_PATH" --dust-threshold 100 --ids 76181,76182 delete-task-in-queue
