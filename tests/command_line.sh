#!/bin/bash
chunkflow generate-tasks -c 0 0 0 -s 0 0 0 -g 1 1 1

chunkflow create-chunk plugin -f median_filter

chunkflow create-chunk -o seg create-chunk -o gt evaluate-segmentation -s seg -g gt

chunkflow log-summary -l tests/data/log

chunkflow create-chunk normalize-intensity
chunkflow create-chunk write-h5 --file-name="/tmp/img.h5" connected-components --threshold 128 write-tif --file-name=/tmp/seg.h5
if test -f /tmp/img.h5 ; then echo "File found"; else exit 1; fi
chunkflow read-h5 --file-name=/tmp/img.h5

chunkflow --dry-run --verbose 1 \
    setup-env -l "gs://my/path" --volume-start 2002 25616 12304 \
    --volume-stop 2068 26128 12816 --max-ram-size 14 --input-patch-size 20 128 128 \
    --output-patch-size 16 96 96 --output-patch-overlap 6 32 32 --channel-num 3 \
    --dtype float32 -m 0 --encoding raw --voxel-size 45 16 16 --max-mip 5

chunkflow \
    create-chunk --size 36 448 448 \
    inference --input-patch-size 20 256 256 --patch-num 2 2 2 \
        --framework "universal" --convnet-model "./examples/inference/universal_identity.py" \
        --batch-size 3

chunkflow \
    create-chunk --size 36 448 448 \
    inference --input-patch-size 20 256 256 \
        --patch-num 2 2 2 --framework identity --batch-size 3 

chunkflow create-chunk --all-zero --size 36 448 448 \
    inference --input-patch-size 20 256 256 --patch-num 2 2 2 \
        --framework identity --batch-size 3 

chunkflow \
    create-chunk --size 36 448 448 --dtype "uint32" \
	connected-components mask-out-objects -d 50 -s "2,3,4"\
	skeletonize --voxel-size 1 1 1 --output-path "file:///tmp/test/skeleton"
