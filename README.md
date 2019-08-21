chunkflow 
----------------------
[![Documentation Status](https://readthedocs.org/projects/pychunkflow/badge/?version=latest)](https://pychunkflow.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/seung-lab/chunkflow.svg?branch=master)](https://travis-ci.org/seung-lab/chunkflow)
[![PyPI version](https://badge.fury.io/py/chunkflow.svg)](https://badge.fury.io/py/chunkflow)
[![Coverage Status](https://coveralls.io/repos/github/seung-lab/chunkflow/badge.svg?branch=master)](https://coveralls.io/github/seung-lab/chunkflow?branch=master)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Chunk operations for large scale 3D image dataset processing

# Motivation
Benefited from the rapid development of microscopy technologies, we can acquire large scale 3D volumetric datasets with both high resolution and large field of view. These 3D image datasets are too big to be processed in a single computer, and distributed processing is required. In most cases, the image dataset could be choped to chunks with/without overlap and distributed to computers for processing. 

Inside each single computer, we perform some operations of the image chunk. The type or order of operations varies according to the image type, quality and application, it is hard to make a universal pipeline for general usage or create specialized pipeline for each usage case. Composing and reusing operators to create customized pipeline easily will facilitate usage. 

Chunkflow provides a framework to perform distributed chunk processing for large scale 3D image dataset. For each task in a single computer, you can compose operators to create pipeline easily for each use case.

## Features
- Composable operators. The chunk operators could be freely composed in commandline for flexible usage.
- Distributed computation in both local and cloud computers. The task scheduling frontend and computationally heavy backend are decoupled using AWS Simple Queue Service. The computational heavy backend could be any computer with internet connection and Amazon Web Services (AWS) authentication.
- All operations support 3D.

## Some Typical Operators
- Convolutional Network Inference. Currently, we support [PyTorch](https://pytorch.org) and [pznet](https://github.com/supersergiy/znnphi_interface)
- Image segmentation using watershed and mean affinity agglomeration.
- Image segmentation using connected component.
- Cutout service. Cutout/save chunk from/to datasets formatted as [neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) using [cloudvolume](https://github.com/seung-lab/cloud-volume)
- Read/write hdf5 and tif files.
- Visualization using [neuroglancer](https://github.com/google/neuroglancer).
- Evaluation of segmentation using rand index and variation of information.

## Terminology
- patch: ndarray as input to ConvNet. Normally it is pretty small due to the limited memory capacity of GPU.
- chunk: ndarray with global offset and arbitrary shape.
- block: the array with a shape and global offset aligned with storage backend. The block could be saved directly to storage backend. The alignment with storage files ensures that there is no writting conflict when saved parallelly.

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
