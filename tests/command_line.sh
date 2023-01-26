#!/bin/bash

echo -e "\ngenerate tasks..."
chunkflow generate-tasks -c 0 0 0 -s 0 0 0 -g 1 1 1

echo -e "\ndownsample image..."
chunkflow create-chunk --dtype uint8 --pattern sin downsample
echo -e "\ndownsample segmentation..."
chunkflow create-chunk --dtype uint32 --pattern sin downsample
# we do not support float32 yet, but will be add later.
# chunkflow create-chunk --dtype float32 --pattern sin downsample


# this example is from [nuclease](https://github.com/janelia-flyem/neuclease)
# echo -e "test nuclease..."
# chunkflow generate-tasks --roi-start 20789 21341 17019 --chunk-size 16 128 128  plugin -f cutout_dvid_label -i bbox -o chunk

#echo -e "\nread chunks using tensorstore."
#chunkflow \
#    generate-tasks --roi-start 20000 20000 20000  --chunk-size 32 32 32 \
#    load-tensorstore -s gs://neuroglancer-janelia-flyem-hemibrain/v1.1/segmentation/ --voxel-size 8 8 8 \

echo -e "\ncreate a hdf5 file, then test the skip-task operator."
chunkflow create-chunk --pattern zero skip-all-zero save-h5 -f /tmp/
# the file already exist, so we'll skip that task
# if the skip is not successful, there is no chunk to write, we'll get an error.
chunkflow create-chunk save-h5 -f /tmp/

echo -e "\ntest skip tasks..."
chunkflow generate-tasks --roi-start 0 0 0 --chunk-size 64 64 64 skip-task --prefix /tmp/ --suffix .h5 
rm /tmp/0-64_0-64_0-64.h5

echo -e "\ntest plugin with arguments..."
chunkflow create-chunk plugin -f median_filter -i chunk -o chunk --args "size=(3,1,1);mode=reflect"

echo -e "\nevaluate segmentation quality..."
chunkflow create-chunk -o seg create-chunk -o gt evaluate-segmentation -s seg -g gt

echo -e "\ngenerate summary of log..."
chunkflow log-summary -l tests/data/log

echo -e "\ntest write png files..."
chunkflow create-chunk save-pngs -o /tmp/pngs; rm -rf /tmp/pngs

echo -e "\ntest normalize intensity..."
chunkflow create-chunk normalize-intensity

echo -e "\nconnected component analysis..."
chunkflow create-chunk save-h5 --file-name="/tmp/img.h5" connected-components --threshold 128 save-h5 --file-name=/tmp/seg.h5
if test -f /tmp/img.h5 ; then echo -e "File found"; else exit 1; fi
chunkflow load-h5 --file-name=/tmp/img.h5

echo -e "\nwrite and read nrrd file."
chunkflow create-chunk save-nrrd -f "/tmp/img.nrrd";
chunkflow load-nrrd -f "/tmp/img.nrrd"; 
rm /tmp/img.nrrd;

echo -e "\nwrite and read tif file."
chunkflow create-chunk save-tif -f "/tmp/img.tif";
chunkflow load-tif -f "/tmp/img.tif"; 
rm /tmp/img.tif;
chunkflow create-chunk --dtype uint16 --pattern sin save-tif -f /tmp/seg.tif
chunkflow load-tif -f /tmp/seg.tif
rm /tmp/seg.tif

echo -e "\ncreate the info file of Neuroglancer Precomputed format."
mkdir /tmp/seg
chunkflow \
    create-chunk --size 128 128 128 --dtype uint32 --pattern sin \
    create-info --voxel-size 8 8 8 --block-size 64 64 64 --output-layer-path file:///tmp/seg/

echo -e "\nwrite image to precomputed volume."
# somehow, we have to separate creation of info and writing out precomputed operation!
chunkflow \
    create-chunk --size 128 128 128 --dtype uint32 --pattern sin \
    save-precomputed --volume-path file:///tmp/seg \
    mesh --voxel-size 8 8 8 --output-format precomputed --output-path file:///tmp/seg 
rm -rf /tmp/seg

echo -e "\ndry run of setup-environment in the cloud."
chunkflow --dry-run --log-level info \
    setup-env -l "gs://my/path" --volume-start 2002 25616 12304 \
        --volume-stop 2068 26128 12816 --max-ram-size 14 \
        --input-patch-size 20 128 128 \
        --output-patch-size 16 96 96 --output-patch-overlap 6 32 32 \
        --channel-num 3 --dtype float32 -m 0 --encoding raw \
        --voxel-size 45 16 16 --max-mip 5

echo -e "\nconvolutional net inference."
chunkflow --log-level debug \
    create-chunk --size 36 448 448 \
    inference --input-patch-size 20 256 256 --patch-num 2 2 2 \
        --framework "universal" \
        --convnet-model "./examples/inference/universal_identity.py" \
        --batch-size 3 #cloud-watch --log-name chunkflow-test

chunkflow \
    create-chunk --size 36 448 448 \
    inference --input-patch-size 20 256 256 \
        --patch-num 2 2 2 --framework identity --batch-size 3 

chunkflow create-chunk --pattern zero --size 36 448 448 \
    inference --input-patch-size 20 256 256 --patch-num 2 2 2 \
        --framework identity --batch-size 3 

echo -e "\nmask out objects and skeletonize."
chunkflow \
    create-chunk --size 36 448 448 --dtype "uint32" \
	connected-components mask-out-objects -d 50 -s "2,3,4"\
	skeletonize --voxel-size 1 1 1 --output-path "file:///tmp/test/skeleton"
