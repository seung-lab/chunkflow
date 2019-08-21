.. _install:

Install
########
Use Docker Image without Installation
======================================
We have built docker images ready to use. You can use docker images directly without installation and configuration. The docker images are in Docker Hub_.

::
    docker pull seunglab/chunkflow

Install from Pypi
==================
.. note::

    We support python version >=3.5

Install the version released in pypi::

   pip3 install chunkflow

Manual Installation
===================
Install the latest from repo::

   git clone https://github.com/seung-lab/chunkflow.git
   cd chunkflow
   pip install -r requirements.txt
   python setup.py install

