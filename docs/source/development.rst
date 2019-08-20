.. _development:

Development
############

Install
**********
Install with development mode (preferably in a python virtual environment)::
   
   git clone https://github.com/seung-lab/chunkflow.git
   cd chunkflow
   pip3 install -r requirements.txt
   python3 setup.py develop

.. note::

    we support python version >=3.5

Release
***********

#. Update version number. The version number is defined in `chunkflow/__version__.py`, increase the version number before releasing. 

#. Create a new release in PyPi.

.. code-block:: console

   python setup.py sdist
   twine upload dist/chunkflow-version.tar.gz

.. note::

    If you would like to include/exclude some files/folders, please edit the MANIFEST.in file.

Build Docker Image
==================

All the docker images are automatically built and is available in the DockerHub_. The ``latest`` tag is the image built from the ``master`` branch. The ``base`` tag is a base ubuntu image, and the ``pytorch`` and ``pznet`` tag contains ``pytorch`` and ``pznet`` inference backends respectively. 

.. _DockerHub: https://hub.docker.com/r/seunglab/chunkflow

You can also manually build docker images locally. The docker files is organized hierarchically. The ``docker/base/Dockerfile`` is a basic one, and the ``docker/inference/pytorch/Dockerfile`` and ``docker/inference/pznet/Dockerfile`` contains pytorch and pznet respectively for ConvNet inference. 

After building the base images, you can start building chunkflow image with different backends. You can just modify the base choice in the Dockerfile and then build it:

.. code-block:: docker

    # backend: base | pytorch | pznet | pytorch-cuda9
    ARG BACKEND=pytorch 

Documentation
***************
We use `Sphinx`_ with `reStructuredText`_ to make documentation. You can make it locally for tests::

   cd docs
   pip3 install -r requirements.txt
   make html

.. _Sphinx: https://www.sphinx-doc.org
.. _reStructuredText: http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html

You can also make other formats, such as pdf and epub. Checkout it out with `make help`.

Add a New Operator
*******************
#. Take a look of the existing simple operators, such as :file:`chunkflow/flow/create_chunk.py`.

#. Create your own operator file in the `flow` folder.

#. Create your own operator class such as `MyOperator`, and this class should inherit from the `.base.OperatorBase`.

#. Define the job of you operator in `__call__(self, chunk, *other_args, **kwargs)`.

#. add the import operator code in :file:`chunkflow/flow/operators/__init__.py`.

#. Add unit test in :file:`tests/flow/`. 

#. Run your unit test by :code:`pytest tests/flow/my_operator.py`
   
#. Add documentation in :ref:`api`.
