chunkflow 
----------------------
[![Documentation Status](https://readthedocs.org/projects/pychunkflow/badge/?version=latest)](https://pychunkflow.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/seung-lab/chunkflow.svg?branch=master)](https://travis-ci.org/seung-lab/chunkflow)
[![PyPI version](https://badge.fury.io/py/chunkflow.svg)](https://badge.fury.io/py/chunkflow)
[![Coverage Status](https://coveralls.io/repos/github/seung-lab/chunkflow/badge.svg?branch=master)](https://coveralls.io/github/seung-lab/chunkflow?branch=master)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Chunk operations for large scale 3D image dataset processing

Perform Convolutional net inference to segment 3D image volume with one command:
```shell
chunkflow read-tif --file-name path/of/image.tif -o image inference --convnet-model path/of/model.py --convnet-weight-path path/of/weight.pt --patch-size 20 256 256 --patch-overlap 4 64 64 --num-output-channels 3 -f pytorch --batch-size 12 --mask-output-chunk -i image -o affs write-h5 -i affs --file-name affs.h5 agglomerate --threshold 0.7 --aff-threshold-low 0.001 --aff-threshold-high 0.9999 -i affs -o seg write-tif -i seg -f seg.tif neuroglancer -c image,affs,seg -p 33333 -v 30 6 6
```
you can see your 3D image and segmentation in [Neuroglancer](https://github.com/google/neuroglancer)!

![Image_Segmentation](https://github.com/seung-lab/chunkflow/blob/master/docs/source/_static/image/image_seg.png)

We have more operators that can be composed flexiblly, checkout our [Documentation](https://pychunkflow.readthedocs.io/en/latest/).

## Features
- **Composable** operators. The chunk operators could be freely composed in commandline for flexible usage.
- **Distributed** computation in both local and cloud computers. The task scheduling frontend and computationally heavy backend are decoupled using AWS Simple Queue Service. The computational heavy backend could be any computer with internet connection and Amazon Web Services (AWS) authentication.
- All operations support **3D** image volumes.

## Some Typical Operators
- Convolutional Network Inference. Currently, we support [PyTorch](https://pytorch.org) and [pznet](https://github.com/supersergiy/znnphi_interface)
- Image segmentation using watershed and mean affinity agglomeration.
- Image segmentation using connected component.
- Cutout service. Cutout/save chunk from/to datasets formatted as [neuroglancer precomputed](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed) using [cloudvolume](https://github.com/seung-lab/cloud-volume)
- Read/write hdf5 and tif files.
- Visualization using [neuroglancer](https://github.com/google/neuroglancer).
- Evaluation of segmentation using rand index and variation of information.

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
