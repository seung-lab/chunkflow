ChangeLog history
=================

# chunkflow 0.5.7 (2019-xx-xx)
## Breaking Changes

## Deprecations 

## Features
- neuroglancer operator works with multiple chunks (https://github.com/seung-lab/chunkflow/pull/123)
- add connected components operator
- add iterative mean agglomeration operator, including watershed computation.

## Bug Fixes 

## Improved Documentation 
- tutorial for cleft and cell boundary detection (https://github.com/seung-lab/chunkflow/pull/123)

# chunkflow 0.2.6 (2019-03-11)
## Breaking Changes

## Deprecations 

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
## Deprecations 

## Features

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

## Deprecations 

## Features
- add batch size option for ConvNet inference
- better memory footprint by deleting the ascontiguousarray
## Bug Fixes 
- the inference framework was hard coded as identity, fixed this bug.
## Improved Documentation 

# chunkflow 0.2.3 (2019-03-05)
## Breaking Changes

## Deprecations 

## Features

## Bug Fixes 

## Improved Documentation
- updated the documentation to focus on chunk operators, not just ConvNet Inference.

# chunkflow 0.2.2 (2019-03-04)
## Breaking Changes

## Deprecations 

## Features
- add option to blackout some sections
- add a new operator: normalize the sections using precomputed histogram
- add typing check for most of the function arguments and kwargs
## Bug Fixes 

# chunkflow 0.2.1 (2019-03-03)
## Breaking Changes

## Deprecations 

## Features
- rename offset_array to chunk
- rename main.py to flow.py to match the package name
## Bug Fixes 

## Improved Documentation


# chunkflow 0.2.0 (2019-03-03)
## Breaking Changes
- added global parameters
- separate out cloud-watch operator
- rename the consume_task.py to main.py for better code meaning
## Deprecations 

## Features
- reuse operators in loop, so we only need to construct operator once.
- add operator name to add more than one operator with same class. For example, in the inference pipeline, we might mask out both image and affinitymap, the two mask operator should have different names. 
## Bug Fixes 

## Improved Documentation 
# chunkflow 0.1.3 (2019-03-01)
## Breaking Changes
- make all operators inherit from a base class
- move all operators to a separate folder for cleaner code.

## Bug Fixes 
- the default offset in read-file operator was not working correctly. 


# chunkflow 0.1.2 (2019-02-26)
## Breaking Changes

## Deprecations 

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

## Deprecations 

## Features
- composable commandline interface. much easier to use and compose operations.

## Bug Fixes 

## Improved Documentation 

# Template 
the following texts are templates for adding change log

# chunkflow 0.1.0 (2019-xx-xx)
## Breaking Changes

## Deprecations 

## Features

## Bug Fixes 

## Improved Documentation 
