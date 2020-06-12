#!/bin/bash

export QUEUE_NAME="chunkflow"
export VISIBILITY_TIMEOUT="3600"
export IMAGE_LAYER_PATH="gs://my/image/path"
export NORMALIZED_IMAGE_LAYER_PATH="gs://my/normalized/image/path"
export IMAGE_MASK_LAYER_PATH="gs://my/image/mask/path"
export AFF_CONVNET_MODEL_FILE="./aff_net.py"
export AFF_CONVNET_WEIGHT_FILE="./aff_net.chkpt"
export SEM_CONVNET_MODEL_FILE="./sem_net.py"
export SEM_CONVNET_WEIGHT_FILE="./sem_net.chkpt"
export AFF_LAYER_PATH="gs://my/affinity/map/path"
export AFF_MASK_LAYER_PATH="gs://my/affinity/map/path"
export SEM_LAYER_PATH="gs://my/semantic/map/path"

export IMAGE_HISTOGRAM_PATH="gs://my/image/histogram/path"

# perform boundary detection and semantic map inference together 
#chunkflow --verbose 2 --mip 1 \
#    fetch-task -r 20 --queue-name="$QUEUE_NAME" --visibility-timeout=$VISIBILITY_TIMEOUT \
#    cutout --volume-path="$IMAGE_LAYER_PATH" --expand-margin-size 4 64 64 --fill-missing -o "image" \
#    normalize-section-contrast -i "image" -o "image" -p $IMAGE_HISTOGRAM_PATH -l 0.01 -u 0.01 \
#    crop-margin -i "image" -o "cropped-image" -m 4 64 64  \
#    save --name "save-img" -i "cropped-image" --volume-path="$NORMALIZED_IMAGE_LAYER_PATH" --nproc 1 \
#    delete-chunk --name "delete-cropped-image" -c "cropped-image" \
#    inference --name "sem-inference" -i "image" --dtype float32 --convnet-model="$SEM_CONVNET_MODEL_FILE" --convnet-weight-path="${SEM_CONVNET_WEIGHT_FILE}" \
#        --input-patch-size 20 256 256 --output-patch-overlap 4 64 64 --num-output-channels 5 --framework='pytorch' --batch-size 1 --patch-num 7 11 11 \
#    channel-voting \
#    save --name "save-sem" --volume-path="$SEM_LAYER_PATH" --nproc 1 \
#    delete-chunk  --name "delete-sem" -c "chunk" \
#    mask --name='mask-image' -i "image" -o "image" --volume-path="$IMAGE_MASK_LAYER_PATH" --mip 4 --fill-missing --inverse \
#    inference --name "aff-inference" -i "image" --mask-myelin-threshold 0.3 --dtype float32 --convnet-model="$AFF_CONVNET_MODEL_FILE" \
#        --convnet-weight-path="${AFF_CONVNET_WEIGHT_FILE}" --input-patch-size 20 256 256 --output-patch-size 20 256 256 \
#        --output-patch-overlap 4 64 64 --output-crop-margin 4 64 64 --num-output-channels 4 --framework='pytorch' --batch-size 1 --patch-num 7 11 11 \
#    delete-chunk --name "delete-img" -c "image" \
#    mask --name='mask-aff' --volume-path="$AFF_MASK_LAYER_PATH" --mip 4 --fill-missing --inverse \
#    save --name "save-aff" --volume-path="$AFF_LAYER_PATH" --upload-log --nproc 1 --create-thumbnail \
#    delete-chunk --name "delete-aff" -c "chunk" \
#    cloud-watch
#    # delete-task-in-queue

# perform boundary detection only
chunkflow --verbose 2 --mip 0 \
    generate-tasks -m 0 --roi-start 17850 165000 175000 \
        --chunk-size 100 1024 1024 -g 1 1 1 \
    cutout --volume-path="$IMAGE_LAYER_PATH" --expand-margin-size 4 64 64 \
        --fill-missing -o "image" \
    normalize-section-contrast -i "image" -o "image" -p $IMAGE_HISTOGRAM_PATH \
        -l 0.01 -u 0.01 \
    inference --name "aff-inference" -i "image" --mask-myelin-threshold 0.3 \
        --dtype float32 --convnet-model="$AFF_CONVNET_MODEL_FILE" \
        --convnet-weight-path="${AFF_CONVNET_WEIGHT_FILE}" \
        --input-patch-size 20 256 256 --output-patch-size 20 256 256 \
        --output-patch-overlap 4 64 64 --output-crop-margin 4 64 64 \
        --num-output-channels 4 --framework='pytorch' --batch-size 1 \
        --mask-output-chunk \
    delete-chunk --name "delete-img" -c "image" \
    write-h5 -f "affinitymap_175000_165000_17850.h5"
