#!/bin/bash
source env.sh
chunkflow setup-env -l ${AFF_PATH} \
    --volume-start ${VOL_START} --volume-stop ${VOL_STOP} \
    --max-ram-size ${MAX_RAM} \
    --input-patch-size 20 128 128 \
    --output-patch-size 16 96 96 --output-patch-overlap 6 32 32 --crop-chunk-margin 6 32 32 \
    --channel-num 3 \
    -m ${AFF_MIP} \
    --thumbnail --thumbnail-mip 5 \
    --voxel-size ${IMAGE_RESOLUTION} \
    --max-mip ${MAX_MIP} \
    -q amqp://172.31.31.249:5672
