#!/bin/bash
source env.sh

if [ -n "$PYTORCH_MODEL_PKG" ]; then
    gsutil cp ${PYTORCH_MODEL_PKG} .
    tar zxvf pytorch-model.tgz -C /root/workspace/chunkflow
    export PYTHONPATH=/root/workspace/chunkflow/pytorch-model:$PYTHONPATH
fi

export PYTHONPATH=/root/workspace/chunkflow/DeepEM:$PYTHONPATH
export PYTHONPATH=/root/workspace/chunkflow/dataprovider3:$PYTHONPATH
export PYTHONPATH=/root/workspace/chunkflow/pytorch-emvision:$PYTHONPATH

chunkflow --mip ${AFF_MIP} \
    fetch-task-kombu -r 5 --queue-name=amqp://172.31.31.249:5672 \
    cutout --mip ${IMAGE_MIP} --volume-path="$IMAGE_PATH" --expand-margin-size ${EXPAND_MARGIN_SIZE} ${FILL_MISSING} \
    ${CONTRAST_NORMALIZATION} \
    ${MASK_IMAGE} \
    inference --name "aff-inference" \
        --convnet-model=/root/workspace/chunkflow/model.py \
        --convnet-weight-path=/root/workspace/chunkflow/model.chkpt \
        --dtype float32 \
        --num-output-channels 3 \
        --input-patch-size 20 128 128 \
        --output-patch-size 16 96 96 \
        --output-patch-overlap 6 32 32 \
        --output-crop-margin 6 32 32 \
        --framework='pytorch' \
        --batch-size 1 \
        --patch-num ${PATCH_NUM} \
    ${MASK_AFF} \
    save --name "save-aff" \
        --volume-path="$AFF_PATH" \
        --upload-log --create-thumbnail \
    delete-task-in-queue-kombu
