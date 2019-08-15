chunkflow 
----------------------
[![Documentation Status](https://readthedocs.org/projects/pychunkflow/badge/?version=latest)](https://pychunkflow.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/seung-lab/chunkflow.svg?branch=master)](https://travis-ci.org/seung-lab/chunkflow)
[![PyPI version](https://badge.fury.io/py/chunkflow.svg)](https://badge.fury.io/py/chunkflow)
[![Coverage Status](https://coveralls.io/repos/github/seung-lab/chunkflow/badge.svg?branch=master)](https://coveralls.io/github/seung-lab/chunkflow?branch=master)

Chunk operations for large scale 3D image dataset processing

# Motivation
Benefited from the rapid development of microscopy technologies, we can acquire large scale 3D volumetric datasets with both high resolution and large field of view. These 3D image datasets are too big to be processed in a single computer, and distributed processing is required. In most cases, the image dataset could be choped to chunks with/without overlap and distributed to computers for processing. This package provide a framework to perform distributed chunk processing for large scale 3D image dataset. For each task in a single machine, it has a few composable chunk operators for flexible real world usage.

## Features
- Composable Commandline interface. The chunk operators could be freely composed in commandline for flexible usage. This is also super useful for tests and experiments.
- Decoupled frontend and backend. The computational heavy backend could be any computer with internet connection and Amazon Web Services (AWS) authentication. 

## Some Typical Operators
- [x] Convolutional Network Inference. Currently, we support [PyTorch](https://pytorch.org) and [pznet](https://github.com/supersergiy/znnphi_interface)
- [x] Task Generator. Fetch task from AWS SQS.
- [x] Cutout service. Cutout chunk from datasets formatted as [neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) using [cloudvolume](https://github.com/seung-lab/cloud-volume)
- [x] Save. Save chunk to neuroglancer precomputed. 
- [x] Save Images. Save chunk as a serials of PNG images in local directory.
- [x] Real File. Read image from hdf5 and tiff files. 
- [x] View. View chunk using cloudvolume viewer.
- [x] Mask. Mask out the chunk using a precomputed dataset.
- [x] Cloud Watch. Realtime speedometer using AWS CloudWatch.

### Use specific GPU device
We can simply set an environment variable to use specific GPU device.

`CUDA_VISIBLE_DEVICES=2 chunkflow`

## Produce tasks to AWS SQS queue
in `bin`, 

`python produce_tasks.py --help`

## Terminology
- patch: ndarray as input to ConvNet. Normally it is pretty small due to the limited memory capacity of GPU.
- chunk: ndarray with global offset and arbitrary shape.
- block: the array with a shape and global offset aligned with storage backend. The block could be saved directly to storage backend. The alignment with storage files ensures that there is no writting conflict when saved parallelly.

# Development
## Create a new release in PyPi 
```
python setup.py sdist
twine upload dist/chunkflow-version.tar.gz
```
# Citation
If you used this tool and is writing a paper, please cite this paper:
```bibtex
@article{wu2019chunkflow,
  title={Chunkflow: Distributed Hybrid Cloud Processing of Large 3D Images by Convolutional Nets},
  author={Wu, Jingpeng and Silversmith, William M and Seung, H Sebastian},
  journal={arXiv preprint arXiv:1904.10489},
  year={2019}
}
```
