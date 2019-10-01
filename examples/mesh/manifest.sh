#!/bin/bash
export PROCESS_NUM=10


seq "$PROCESS_NUM" | parallel -j "$PROCESS_NUM" --delay 20 --ungroup echo Starting worker {}\; chunkflow mesh-manifest -v gs://my/dataset/path --prefix {}
