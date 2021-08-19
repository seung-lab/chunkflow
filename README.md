![alt text](https://github.com/seung-lab/chunkflow/blob/master/docs/logo/RGB_web/Chunkflow_logo_RBG.jpg?raw=true)
----------------------
[![Documentation Status](https://readthedocs.org/projects/pychunkflow/badge/?version=latest)](https://pychunkflow.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/chunkflow.svg)](https://badge.fury.io/py/chunkflow)
[![Coverage Status](https://coveralls.io/repos/github/seung-lab/chunkflow/badge.svg?branch=master)](https://coveralls.io/github/seung-lab/chunkflow?branch=master)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docker Hub](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/seunglab/chunkflow)
[![Twitter URL](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fjingpeng_wu)](https://twitter.com/jingpeng_wu)
<!---[![Docker Build Status](https://img.shields.io/docker/cloud/build/seunglab/chunkflow.svg)]#(https://hub.docker.com/r/seunglab/chunkflow)--->
<!-- [![Build Status](https://travis-ci.org/seung-lab/chunkflow.svg?branch=master)](https://travis-ci.org/seung-lab/chunkflow) -->

## Features
- **Composable** operators. The chunk operators could be composed in commandline for flexible usage.
- **Hybrid Cloud Distributed** computation in both local and cloud computers. The task scheduling frontend and computationally heavy backend are decoupled using AWS Simple Queue Service. The backend could be any computer with internet connection and cloud authentication.
- **Petabyte** scale. We have used chunkflow to output over eighteen petabyte images and scaled up to 3600 nodes with NVIDIA GPUs across three regions in [Google Cloud](https://cloud.google.com/), and chunkflow is still reliable.
- Operators work with **3D** image volumes.
- You can **plugin** your own code as an operator.

Check out the [Documentation](https://pychunkflow.readthedocs.io/en/latest/index.html) for [installation](https://pychunkflow.readthedocs.io/en/latest/install.html) and [usage](https://pychunkflow.readthedocs.io/en/latest/tutorial.html). Try it out by following the [tutorial](https://pychunkflow.readthedocs.io/en/latest/tutorial.html). 

## Image Segmentation Example
Perform Convolutional net inference to segment 3D image volume with one single command!

```shell
#!/bin/bash

chunkflow \
    read-tif --file-name path/of/image.tif -o image \
    inference --convnet-model path/of/model.py --convnet-weight-path path/of/weight.pt \
        --input-patch-size 20 256 256 --output-patch-overlap 4 64 64 --num-output-channels 3 \
        -f pytorch --batch-size 12 --mask-output-chunk -i image -o affs \
    plugin -f agglomerate --threshold 0.7 --aff-threshold-low 0.001 --aff-threshold-high 0.9999 -i affs -o seg \
    neuroglancer -i image,affs,seg -p 33333 -v 30 6 6
```
you can see your 3D image and segmentation directly in [Neuroglancer](https://github.com/google/neuroglancer)!

![Image_Segmentation](https://github.com/seung-lab/chunkflow/blob/master/docs/source/_static/image/image_seg.png)

## Operators
After installation, You can simply type `chunkflow` and it will list all the operators with help message. We keep adding new operators and will keep it update here. For the detailed usage, please checkout our [Documentation](https://pychunkflow.readthedocs.io/en/latest/).

| Operator Name   | Function |
| --------------- | -------- |
| aggregate-skeleton-fragments| Merge skeleton fragments from chunks |
| channel-voting  | Vote across channels of semantic map |
| cloud-watch     | Realtime speedometer in AWS CloudWatch |
| connected-components | Threshold the boundary map to get a segmentation |
| copy-var        | Copy a variable to a new name |
| create-chunk    | Create a fake chunk for easy test |
| create-info     | Create info file of Neuroglancer Precomputed volume |
| crop-margin     | Crop the margin of a chunk |
| delete-chunk    | Delete chunk in task to reduce RAM requirement |
| delete-task-in-queue | Delete the task in AWS SQS queue |
| downsample-upload | Downsample the chunk hierarchically and upload to volume |
| evaluate-segmentation | Compare segmentation chunks |
| fetch-task-from-file | Fetch task from a file |
| fetch-task-from-sqs | Fetch task from AWS SQS queue one by one |
| generate-tasks  | Generate tasks one by one |
| gaussian-filter | 2D Gaussian blurring operated in-place |
| inference       | Convolutional net inference |
| log-summary     | Summary of logs |
| mask            | Black out the chunk based on another mask chunk |
| mask-out-objects| Mask out selected or small objects |
| mesh            | Build 3D meshes from segmentation chunk |
| mesh-manifest   | Collect mesh fragments for object |
| neuroglancer    | Visualize chunks using neuroglancer |
| normalize-contrast-nkem | Normalize image contrast using histograms |
| normalize-intensity | Normalize image intensity to -1:1 |
| normalize-section-shang | Normalization algorithm created by Shang |
| plugin          | Import local code as a customized operator. |
| quantize        | Quantize the affinity map |
| read-h5         | Read HDF5 files |
| read-json       | Read JSON files |
| read-pngs       | Read png files |
| read-precomputed| Cutout chunk from a local/cloud storage volume |
| read-tif        | Read TIFF files |
| read-nrrd       | Read NRRD files |
| remap-segmentation | Renumber a serials of segmentation chunks |
| setup-env       | Prepare storage infor files and produce tasks |
| skeletonize     | Create centerlines of objects in a segmentation chunk |
| skip-task  	  | If a result file already exists, skip this task |
| skip-all-zero   | If a chunk has all zero, skip this task |	
| threshold       | Use a threshold to segment the probability map |
| view            | Another chunk viewer in browser using CloudVolume |
| write-h5        | Write chunk as HDF5 file |
| write-pngs      | Save chunk as a serials of png files |
| write-precomputed| Save chunk to local/cloud storage volume |
| write-tif       | Write chunk as TIFF file |
| write-nrrd      | Write chunk as NRRD file |



## Reference
We have a [paper](https://www.nature.com/articles/s41592-021-01088-5) for this repo: 
```bibtex

@article{wu_chunkflow_2021,
	title = {Chunkflow: hybrid cloud processing of large {3D} images by convolutional nets},
	issn = {1548-7105},
	shorttitle = {Chunkflow},
	url = {https://www.nature.com/articles/s41592-021-01088-5},
	doi = {10.1038/s41592-021-01088-5},
	journal = {Nature Methods},
	author = {Wu, Jingpeng and Silversmith, William M. and Lee, Kisuk and Seung, H. Sebastian},
	year = {2021},
	pages = {1--2}
}
```
