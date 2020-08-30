ChangeLog history
=================
# chunkflow 1.0.1 (2020-xx-xx)
## Breaking Changes
- renamed custom-operator to plugin
## Deprecations 

## Features
- a plugin system with a median filter example

## Bug Fixes 
- cutout whole volume in default

## Improved Documentation 

# chunkflow 1.0.0 (2020-08-14)
## Breaking Changes
- rename inference backend `general` to `universal`

## Deprecations 

## Features
- threshold operator

## Bug Fixes 

## Improved Documentation 


# chunkflow 0.6.3 (2020-03-23)
## Breaking Changes
- remove convnet inference backend `pytorch-multitask` since it could be included by the `pytorch` backend.
## Deprecations 

## Features
- a general convnet inference backend
- support combined convnet inference including semantic and affinity map inference.
- all-zero option for create-chunk operator and inference test in travis
- new operator to delete chunk for releasing memory

## Bug Fixes 
- log-summary operator will work for combined inference

## Improved Documentation 
- added corresponding documentation
- added new operator 

# chunkflow 0.6.2 (2020-03-06)
## Breaking Changes
- make `verbose` a integer rather than boolean number for variation of verbosity.

## Features
- add setup-env operator to automatically compute patch number, cloud storage block size and other metadata. ingest tasks into AWS SQS queue. After this operation, you are ready to launch workers!
- support cropped output patch size for inference
- refactored normalize section contrast operator to make it faster and easier to use. We precompute a lookup table to avoid redundent computation of the voxel mapping.
- avoid creating a mask buffer by directly applying the high-mip mask to low-mip chunk

## Bug Fixes 
- fix a typo of thumbnail_mip

## Improved Documentation 
- add more complex production inference example

# chunkflow 0.5.7 (2019-02-02)
## Breaking Changes
- remove c++ compilation module

## Features
- neuroglancer operator works with multiple chunks (https://github.com/seung-lab/chunkflow/pull/123)
- add connected components operator
- add iterative mean agglomeration operator, including watershed computation.

## Improved Documentation 
- tutorial for cleft and cell boundary detection (https://github.com/seung-lab/chunkflow/pull/123)

# chunkflow 0.2.6 (2019-03-11)

## Features
- a new operator called `save-images` to save chunk as a serials of PNG images.
- add log directory parameter for benchmark script.
## Bug Fixes 
- queue becomes None
- pznet was not working.
- fix Dockerfile for pznet build (not sure whether breaks PyTorch or not)

## Improved Documentation 
- add kubernetes documentation to find monitor of operations per second and bandwidth usage.

# chunkflow 0.2.5 (2019-03-08)
## Breaking Changes
- change the mask command parameter `mask-mip` to `mip`

## Bug Fixes 
- fix log uploading and cloud watch
- fix the misplaced log in task, which will make the inference log not working
- fix blackout section bug. Previous implementation will have indexing error while blacking out z outside of chunk.
## Improved Documentation
- clarify patch, chunk and block.
- add realistic example for ConvNet inference.

# chunkflow 0.2.4 (2019-03-07)
## Breaking Changes
- merge operator upload-log to save since we normally just put log inside saved volume.

## Features
- add batch size option for ConvNet inference
- better memory footprint by deleting the ascontiguousarray
## Bug Fixes 
- the inference framework was hard coded as identity, fixed this bug.

# chunkflow 0.2.3 (2019-03-05)
## Improved Documentation
- updated the documentation to focus on chunk operators, not just ConvNet Inference.

# chunkflow 0.2.2 (2019-03-04)
## Features
- add option to blackout some sections
- add a new operator: normalize the sections using precomputed histogram
- add typing check for most of the function arguments and kwargs

# chunkflow 0.2.1 (2019-03-03)
## Features
- rename offset_array to chunk
- rename main.py to flow.py to match the package name

# chunkflow 0.2.0 (2019-03-03)
## Breaking Changes
- added global parameters
- separate out cloud-watch operator
- rename the consume_task.py to main.py for better code meaning

## Features
- reuse operators in loop, so we only need to construct operator once.
- add operator name to add more than one operator with same class. For example, in the inference pipeline, we might mask out both image and affinitymap, the two mask operator should have different names. 

# chunkflow 0.1.3 (2019-03-01)
## Breaking Changes
- make all operators inherit from a base class
- move all operators to a separate folder for cleaner code.

## Bug Fixes 
- the default offset in read-file operator was not working correctly. 


# chunkflow 0.1.2 (2019-02-26)
## Features
- processors for read/write hdf5 files 
- processor for create fake image chunk for tests
- travis test for chunkflow commandline interface

## Bug Fixes 
- fix a bug of OffsetArray. The attribute changes after numpy operations. There is still some operation will change the attribute, will fix later.

## Improved Documentation 
- add documentation to release pypi package

# chunkflow 0.1.0 (2019-02-25)
## Breaking Changes
- decompose all the code to individual functions to process chunks 
- the command line usage is completely different 
- no multiprocessing internally. it has to be implemented in shell script. 

## Features
- composable commandline interface. much easier to use and compose operations.


# Template 
the following texts are templates for adding change log

# chunkflow 1.1.0 (2020-xx-xx)
## Breaking Changes

## Deprecations 

## Features

## Bug Fixes 

## Improved Documentation 
