# Introduction
3D convnet becomes more and more popular to segment 3D images. Since single machine and limited memory including GPU memory, a large dataset can not fit in for one-time convnet forwardpass especially for large complex networks. Hence, convnet inference should be decomposed to multiple patches and then stitch the patches together. The patches could be well distributed across machines utilizing the data level parallelism. However, there normally exist boundary effect of each patch since there just no enough context information for the boundary voxels. To reduce the boundary effect, the patches could be blended with some overlap. Overlap and blending could be easily handled in a single shared-memory machine, but not for distributed computation for terabyte or petabyte scale inference. Naive cropping of chunk boundary produced in each machine will make the output globally inconsistent unless the cropping size is half of the patch size which has too much overhead. This package was made to solve this problem.

## Terminology
- patch: the input/output 3D/4D array for convnet with typical size like 32x256x256.
- chunk: the input/output 3D/4D array after blending in each machine with typical size like 116x1216x1216.
- block: the final main output array of each machine which should be aligned with storage backend such as neuroglancer precomputed. The typical size is like 112x1152x1152.

The strategy is decomposing the inference to two steps with some dependency. 

## Inference and Donate
For each input chunk, 
- run convnet with blending
- save core valid chunk to final output place, such as google cloud storage managed using [cloudvolume](https://github.com/seung-lab/cloud-volume).
- donate all the overlapping chunks to a shared storage, such as cloud storage.

The regions need blending from neighboring chunks will be handled in the next step.

Take 1D case for example, the left and right hand side with overlapping region will be donated to shared storage with unique filename. The filename was encoded as `block-range%chunk-range`, such as `0-1152_0-1152_0-112%0-64_0-64_0-4`.

## Receive and Blend
For each output block:
- Fetch the overlapping chunks for blending.
- Blend the chunks together. The operation is just addition since all the patches were already normalized.
- Write back to output storage. This have to take care of the alignment of blockes in cloud storage.

# Usage

## Inference and Donate
run convnet inference and donate the extra chunks
check out the options with:
`python inference_and_donate.py -h`

## Receive and Blend 
receive shared chunks and blend them together

checkout the options with:
`python receive_and_blend.py -h`

### Use with backend of PZNet 

#### Compile PZNet

### Use with backend of PyTorch 
- required pytorch version 0.4.x


