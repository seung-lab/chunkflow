.. chunkflow documentation master file, created by
   sphinx-quickstart on Thu Jul 25 17:29:21 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to chunkflow's documentation!
######################################
Motivation
===========
Benefited from the rapid development of microscopy technologies, we can acquire large scale 3D volumetric datasets with both high resolution and large field of view. These 3D image datasets are too big to be processed in a single computer, and distributed processing is required. In most cases, the image dataset could be choped to chunks with/without overlap and distributed to computers for processing. 

Inside each single computer, we perform some operations of the image chunk. The type or order of operations varies according to the image type, quality and application, it is hard to make a universal pipeline for general usage or create specialized pipeline for each usage case. Composing and reusing operators to create customized pipeline easily will facilitate usage. 

Chunkflow provides a framework to perform distributed chunk processing for large scale 3D image dataset. For each task in a single computer, you can compose operators to create pipeline easily for each use case.

Terminology
============
- patch: ndarray as input to ConvNet. Normally it is pretty small due to the limited memory capacity of GPU.
- chunk: ndarray with global offset and arbitrary shape.
- block: the array with a shape and global offset aligned with storage backend. The block could be saved directly to storage backend. The alignment with storage files ensures that there is no writting conflict when saved parallelly.



.. toctree::
  :maxdepth: 2
  
  install
  tutorial
  api
  development

Indices and tables
###################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
