chunkflow 
----------------------
[![Documentation Status](https://readthedocs.org/projects/pychunkflow/badge/?version=latest)](https://pychunkflow.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/seung-lab/chunkflow.svg?branch=master)](https://travis-ci.org/seung-lab/chunkflow)
[![PyPI version](https://badge.fury.io/py/chunkflow.svg)](https://badge.fury.io/py/chunkflow)
[![Coverage Status](https://coveralls.io/repos/github/seung-lab/chunkflow/badge.svg?branch=master)](https://coveralls.io/github/seung-lab/chunkflow?branch=master)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docker Hub](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/seunglab/chunkflow)
[![Twitter URL](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fjingpeng_wu)](https://twitter.com/jingpeng_wu)
<!---[![Docker Build Status](https://img.shields.io/docker/cloud/build/seunglab/chunkflow.svg)]#(https://hub.docker.com/r/seunglab/chunkflow)--->

Perform Convolutional net inference to segment 3D image volume with one single command!

```shell
chunkflow read-tif --file-name path/of/image.tif -o image inference --convnet-model path/of/model.py --convnet-weight-path path/of/weight.pt --input-patch-size 20 256 256 --output-patch-overlap 4 64 64 --num-output-channels 3 -f pytorch --batch-size 12 --mask-output-chunk -i image -o affs write-h5 -i affs --file-name affs.h5 agglomerate --threshold 0.7 --aff-threshold-low 0.001 --aff-threshold-high 0.9999 -i affs -o seg write-tif -i seg -f seg.tif neuroglancer -c image,affs,seg -p 33333 -v 30 6 6
```
you can see your 3D image and segmentation directly in [Neuroglancer](https://github.com/google/neuroglancer)!

![Image_Segmentation](https://github.com/seung-lab/chunkflow/blob/master/docs/source/_static/image/image_seg.png)



## Features
- **Composable** operators. The chunk operators could be freely composed in commandline for flexible usage.
- **Hybrid Cloud Distributed** computation in both local and cloud computers. The task scheduling frontend and computationally heavy backend are decoupled using AWS Simple Queue Service. The computational heavy backend could be any computer with internet connection and Amazon Web Services (AWS) authentication.
- All operations support **3D** image volumes.

## Operators
After installation, You can simply type `chunkflow` and it will list all the operators with help message. We list the available operators here. We keep adding new operators and will keep it update here. For the detailed usage, please checkout our [Documentation](https://pychunkflow.readthedocs.io/en/latest/).

| Operator Name   | Function |
| --------------- | -------- |
| agglomerate     | Watershed and agglomeration to segment affinity map |
| channel-voting  | Vote across channels of semantic map |
| cloud-watch     | Realtime speedometer in AWS CloudWatch |
| connected-components | Threshold the boundary map to get a segmentation |
| copy-var        | Copy a variable to a new name |
| create-chunk    | Create a fake chunk for easy test |
| crop-margin     | Crop the margin of a chunk |
| custom-operator | Import local code as a customized operator |
| cutout          | Cutout chunk from a local/cloud storage volume |
| delete-chunk    | Delete chunk in task to reduce RAM requirement |
| delete-task-in-queue | Delete the task in AWS SQS queue |
| downsample-upload | Downsample the chunk hierarchically and upload to volume |
| evaluate-segmentation | Compare segmentation chunks |
| fetch-task      | Fetch task from AWS SQS queue one by one |
| generate-tasks  | Generate tasks one by one |
| inference       | Convolutional net inference |
| log-summary     | Summary of logs |
| mask            | Black out the chunk based on another mask chunk |
| mesh            | Build 3D meshes from segmentation chunk |
| mesh-manifest   | Collect mesh fragments for object |
| neuroglancer    | Visualize chunks using neuroglancer |
| normalize-section-contrast | Normalize image contrast |
| normalize-section-shang | Normalization algorithm created by Shang |
| quantize        | Quantize the affinity map |
| read-h5         | Read HDF5 files |
| read-tif        | Read TIFF files |
| save            | Save chunk to local/cloud storage volume |
| save-pngs       | Save chunk as a serials of png files |
| setup-env       | Prepare storage infor files and produce tasks |
| view            | Another chunk viewer in browser using CloudVolume |
| write-h5        | Write chunk as HDF5 file |
| write-tif       | Write chunk as TIFF file |


## Reference
We have a [paper](https://arxiv.org/abs/1904.10489) of this repo: 
```bibtex
@article{wu2019chunkflow,
  title={Chunkflow: Distributed Hybrid Cloud Processing of Large 3D Images by Convolutional Nets},
  author={Wu, Jingpeng and Silversmith, William M and Seung, H Sebastian},
  journal={arXiv preprint arXiv:1904.10489},
  year={2019}
}
```
