.. _install:

Install
########
Install from Pypi
==================
.. note::

    We support python version >=3.6 in Ubuntu >= 16.04. It should work with other Linux version too. If you need support of other python version or OS, please create an issue in github. 

Install the version released in pypi::

   pip3 install chunkflow

.. note::

   You need g++>=4.8 since some dependent packages need to compile c++ code.

Manual Installation
===================
Install the latest from repo::

   git clone https://github.com/seung-lab/chunkflow.git
   cd chunkflow
   pip install -r requirements.txt
   python setup.py install

.. note::
   
   You need g++>=4.8 since some dependent packages need to compile c++ code.

Use Docker Image without Installation
======================================
We have built `docker
<https://docs.docker.com/>`_ images ready to use. You can use docker images directly without installation and configuration. The docker images are in `Docker Hub
<https://hub.docker.com/r/seunglab/chunkflow>`_.

You can get the image by:

.. code-block::

    docker pull seunglab/chunkflow


