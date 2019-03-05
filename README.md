chunkflow 
----------------------
[![Build Status](https://travis-ci.org/seung-lab/chunkflow.svg?branch=master)](https://travis-ci.org/seung-lab/chunkflow)
[![PyPI version](https://badge.fury.io/py/chunkflow.svg)](https://badge.fury.io/py/chunkflow)
[![Coverage Status](https://coveralls.io/repos/github/seung-lab/chunkflow/badge.svg?branch=master)](https://coveralls.io/github/seung-lab/chunkflow?branch=master)

Chunk operations for large scale 3D image dataset processing

# Introduction
3D image dataset could be too large to be processed in a single computer, and distributed processing was required. In most cases, the image dataset could be choped to chunks and distributed to computers for processing. This package provide a framework to perform distributed chunk processing. 

## Features
- Decoupled frontend and backend. The computational heavy backend could be any computer with internet connection and Amazon Web Services (AWS) authentication. 
- Composable Commandline interface. The chunk operators could be freely composed in commandline for flexible usage. This is also super useful for tests and experiments.

# Usage

## Installation
This package was registered in PyPi, just run a simple command to install:
```
pip install chunkflow
```

## Get Help
`chunkflow --help`

get help for commands: `chunkflow command --help`

## Examples
The commands could be composed and used flexiblly. The first command should be a generator though.
```
chunkflow create-chunk view
chunkflow create-chunk 
```

## Some Typical Operators
- [x] Convolutional Network Inference. Currently, we support [PyTorch](https://pytorch.org) and [pznet](https://github.com/supersergiy/znnphi_interface)
- [x] Task Generator. Fetch task from AWS SQS.
- [x] Cutout service. Cutout chunk from datasets formatted as [neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) using [cloudvolume](https://github.com/seung-lab/cloud-volume)
- [x] Save. Save chunk to neuroglancer precomputed. 
- [x] Real File. Read image from hdf5 and tiff files. 
- [x] Upload Log. upload log information to storage.
- [x] View. View chunk using cloudvolume viewer.
- [x] Mask. Mask out the chunk using a precomputed dataset.
- [x] Cloud Watch. Realtime speedometer using AWS CloudWatch.


## Produce tasks to AWS SQS queue
in `bin`, 

`python produce_tasks.py --help`

## Terminology
- patch: the input/output 3D/4D array for convnet with typical size like 32x256x256.
- chunk: the input/output 3D/4D array after blending in each machine with typical size like 116x1216x1216.
- block: the final main output array of each machine which should be aligned with storage backend such as [neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed). The typical size is like 112x1152x1152.

### Use specific GPU device
We can simply set an environment variable to use specific GPU device.

`CUDA_VISIBLE_DEVICES=2 python consume_tasks.py `

# Development
## Create a new release in PyPi 
```
python setup.py bdist_wheel --universal
twine upload dist/my-new-wheel
```

## Add a new operator
To be added.
