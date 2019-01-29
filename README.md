chunkflow 
----------------------
[![Build Status](https://travis-ci.org/seung-lab/chunkflow.svg?branch=master)](https://travis-ci.org/seung-lab/chunkflow)

patch by patch convolutional network inference with multiple frameworks including pytorch and pznet. 

# Introduction
3D convnet is state of the art approach to segment 3D images. Since single machine has limited computational power and RAM capacity, a large dataset can not fit in for one-time convnet inference especially for large complex networks. Hence, convnet inference should be decomposed to multiple patches and then stitch the patches together. The patches could be well distributed across machines utilizing the data level parallelism. However, there normally exist boundary effect of each patch since the image context around boundary voxels is missing. To reduce the boundary effect, the patches could be blended with some overlap. Overlap and blending could be easily handled in a single shared-memory machine, but not for distributed computation for terabyte or petabyte scale inference. This package was made to solve this problem. The solution is simply cropping the surrounding regions and stitch them together. 

The boundary effect due to the cropping depends on the cropping size. If the cropping size is half of the patch size, there will be no boundary effect, but there is a lot of waste. In practise, we found that about 20%-25% of patch size is reasonably good enough. 

## Supported backends 
- [x] pytorch 
- [x] pznet 

## Terminology
- patch: the input/output 3D/4D array for convnet with typical size like 32x256x256.
- chunk: the input/output 3D/4D array after blending in each machine with typical size like 116x1216x1216.
- block: the final main output array of each machine which should be aligned with storage backend such as [neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed). The typical size is like 112x1152x1152.

# Usage

## Produce tasks
in `scripts`, 

`python produce_tasks.py --help`

## launch worker to consume tasks  
in the `scripts` folder,

`python consume_tasks.py --help`

### use specific GPU device
we can simply set an environment variable to use specific GPU device.

`CUDA_VISIBLE_DEVICES=2 python consume_tasks.py `

