#!/bin/bash
chunkflow generate-tasks -c 0 0 0 -s 0 0 0 -g 1 1 1

# this example is from [nuclease](https://github.com/janelia-flyem/neuclease)
chunkflow generate-tasks --roi-start 20789 21341 17019 --chunk-size 16 128 128  plugin -f cutout_dvid_label -i bbox -o chunk

chunkflow create-chunk --all-zero skip-all-zero write-h5 -f /tmp/

# create a hdf5 file, then test the skip-task operator
# the file already exist, so we'll skip that task
# if the skip is not successful, there is no chunk to write, we'll get an error.
chunkflow create-chunk write-h5 -f /tmp/
chunkflow generate-tasks --roi-start 0 0 0 --chunk-size 64 64 64 skip-task --pre /tmp/ --post .h5 write-h5 -f /tmp/
rm /tmp/0-64_0-64_0-64.h5

chunkflow create-chunk plugin -f median_filter -i chunk -o chunk

chunkflow create-chunk -o seg create-chunk -o gt evaluate-segmentation -s seg -g gt

chunkflow log-summary -l tests/data/log

chunkflow create-chunk write-pngs -o /tmp/pngs; rm -rf /tmp/pngs

chunkflow create-chunk normalize-intensity
chunkflow create-chunk write-h5 --file-name="/tmp/img.h5" connected-components --threshold 128 write-h5 --file-name=/tmp/seg.h5
if test -f /tmp/img.h5 ; then echo "File found"; else exit 1; fi
chunkflow read-h5 --file-name=/tmp/img.h5

chunkflow create-chunk write-nrrd -f "/tmp/img.nrrd";
chunkflow read-nrrd -f "/tmp/img.nrrd"; 
rm /tmp/img.nrrd;

chunkflow create-chunk write-tif -f "/tmp/img.tif";
chunkflow read-tif -f "/tmp/img.tif"; 
rm /tmp/img.tif;

mkdir /tmp/seg
chunkflow \
    create-chunk --size 128 128 128 --dtype uint32 --not-all-zero \
    create-info --voxel-size 8 8 8 --block-size 64 64 64 --output-layer-path file:///tmp/seg/ 

# somehow, we have to separate creation of info and writing out precomputed operation!
chunkflow \
    create-chunk --size 128 128 128 --dtype uint32 --not-all-zero \
    write-precomputed --volume-path file:///tmp/seg \
    mesh --voxel-size 8 8 8 --output-format precomputed --output-path file:///tmp/seg 
rm -rf /tmp/seg

chunkflow --dry-run --log-level info\
    setup-env -l "gs://my/path" --volume-start 2002 25616 12304 \
    --volume-stop 2068 26128 12816 --max-ram-size 14 --input-patch-size 20 128 128 \
    --output-patch-size 16 96 96 --output-patch-overlap 6 32 32 --channel-num 3 \
    --dtype float32 -m 0 --encoding raw --voxel-size 45 16 16 --max-mip 5

chunkflow create-chunk --size 36 448 448 inference --input-patch-size 20 256 256 --patch-num 2 2 2 \
    --framework "universal" --convnet-model "./examples/inference/universal_identity.py" \
    --batch-size 3 #cloud-watch --log-name chunkflow-test
    
chunkflow --log-level debug\
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
