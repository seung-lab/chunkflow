ChangeLog history
=================
# chunkflow 0.1.0 (2019-02-25)
## Breaking Changes
- make all operators inherit from a base class
- move all operators to a separate folder for cleaner code.
## Deprecations 

## Features

## Bug Fixes 
- fix default global offset not working in read file operator
## Improved Documentation 

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

# chunkflow 0.1.0 (2019-02-25)
## Breaking Changes

## Deprecations 

## Features

## Bug Fixes 

## Improved Documentation 
