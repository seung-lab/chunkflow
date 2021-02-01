ChangeLog history
==================

# chunkflow 1.0.6 (2021-xx-xx)
## Breaking Changes

## Deprecations 

## Features
- new operators to read and write NRRD file
## Bug Fixes 

## Improved Documentation
 
# chunkflow 1.0.5 (2021-01-29)
## Breaking Changes
- use python logging module instead of verbose parameter
- changed the `plugin` operator parameter input and output names. So it can accept both chunk and other data, such as bounding box.
- the default downsampling factor changes from (1,2,2) to (2,2,2)
- default input to plugin changed from chunk to None
## Deprecations 

## Features
- quit neuroglancer by enter q and return.
- work with Flatiron disBatch.
- a new operator remap to renumber a serials of segmentation chunks.
- shard meshing. It is not validated yet. The data is written, but Neuroglancer is still not showing them. There is something like manifest missing.
- support downsample in Z
- support disbatch in manifest operator
- add voxel size of chunk
- automatically use chunk voxel size in neuroglancer operator
- mask out a chunk with smaller voxel size. The voxel size should be divisible though.
- plugin with bounding box and argument
- more options for generate-tasks operator
## Bug Fixes 
- fix manifest by updating cloud storage to CloudFiles
- fix read-h5 operator when only cutout size is provided
- fix grid size bug of generating tasks
## Improved Documentation 

# chunkflow 1.0.4 (2020-11-16)
## Breaking Changes
- read-pngs parameter name change.
- rename several operators to make them more consistent: cutout --> read-precomputed, save --> write-precomputed, save-pngs --> write-pngs

## Deprecations 

## Features
- add a plugin to inverse image / affinitymap intensity

## Bug Fixes 

## Improved Documentation 

# chunkflow 1.0.3 (2020-10-29)
## Breaking Changes
- replace global_offset to voxel_offset to be consistent with neuroglancer info 
- change operator name `fetch-task` to `fetch-task-from-sqs` to be more specific.

## Features
- output tasks as a numpy array and save as npy file.
- work with slurm cluster
- fetch task from numpy array

## Bug Fixes 
- fetch task from numpy array

## Improved Documentation 

# chunkflow 1.0.2 (2020-10-01)
## Features
- hdf5 file with cutout range

## Bug Fixes 
- fix neuroglancer visualization for segmentation.

## Improved Documentation 

# chunkflow 1.0.1 (2020-09-01)
## Breaking Changes
- renamed custom-operator to plugin
## Deprecations 

## Features
- new operator: normalize-intensity
- a plugin system with a median filter example
- new operator: normalize-intensity
- support cutout from hdf5 file

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

# chunkflow 1.1.0 (2021-xx-xx)
## Breaking Changes

## Deprecations 

## Features

## Bug Fixes 

## Improved Documentation 
